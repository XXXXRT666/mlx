// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/compile_impl.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

namespace {

detail::CompileGraph optimize(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    bool shapeless = false,
    CompileMode mode = CompileMode::no_fuse) {
  eval(inputs);
  detail::CompileGraph graph(inputs, outputs, {});
  detail::compile_optimize(
      graph,
      detail::CompileContext(shapeless, detail::CompileTarget::Compile, mode));
  graph.validate();
  return graph;
}

} // namespace

TEST_CASE("test compile reorder view chains use linear graph edits") {
  auto x = reshape(arange(8, float32), {2, 4});
  eval(x);
  auto output = x;
  constexpr int layers = 64;
  for (int i = 0; i < layers; ++i) {
    output = reshape(sin(reshape(output, {4, 2})), {2, 4});
  }
  detail::CompileGraph graph({x}, {output}, {});
  const detail::PassSpec passes[] = {
      {"reorder_pointwise_views",
       detail::reorder_pointwise_views,
       detail::PassScope::CompileAndExport}};
  detail::run_compile_passes(
      graph,
      detail::CompileContext(
          false, detail::CompileTarget::Compile, CompileMode::no_fuse),
      "canonicalize",
      passes);
  graph.validate();
  CHECK_LE(graph.rewrite_count(), 2 * layers);
  CHECK(array_equal(graph.outputs()[0], output).item<bool>());
}

TEST_CASE("test compile reorder views exposes pointwise fusion") {
  const std::function<array(const array&)> views[] = {
      [](const array& a) { return reshape(a, {3, 2}); },
      [](const array& a) { return transpose(a); },
      [](const array& a) { return flatten(a); },
      [](const array& a) { return expand_dims(a, 0); },
      [](const array& a) { return squeeze(a, 1); },
      [](const array& a) { return unflatten(a, 2, {1, 3}); },
      [](const array& a) { return broadcast_to(a, {4, 2, 1, 3}); },
      [](const array& a) { return repeat(a, 3, 1); },
      [](const array& a) { return tile(a, {2, 1, 2}); }};
  auto x = reshape(arange(6, float32), {2, 1, 3});
  for (const auto& view : views) {
    auto expected = negative(view(abs(x)));
    auto graph = optimize({x}, {expected}, false, CompileMode::enabled);
    auto node = graph.outputs()[0];
    CHECK_EQ(node.shape(), expected.shape());
    while (node.has_primitive() && node.inputs().size() == 1 &&
           !dynamic_cast<Compiled*>(&node.primitive())) {
      node = node.inputs()[0];
    }
    REQUIRE(node.has_primitive());
    CHECK(dynamic_cast<Compiled*>(&node.primitive()));
    CHECK(array_equal(graph.outputs()[0], expected).item<bool>());
  }
}

TEST_CASE("test compile reorder binary and ternary views") {
  auto x = reshape(arange(6, float32), {2, 3});
  auto y = negative(x);
  auto mask = greater(x, array(2));
  for (auto output :
       {transpose(x) + transpose(y),
        where(transpose(mask), transpose(x), transpose(y)),
        array(2.0f) + reshape(x, {3, 2})}) {
    auto graph = optimize({x, y, mask}, {output, output});
    const auto& root = graph.outputs()[0];
    CHECK_EQ(root.id(), graph.outputs()[1].id());
    CHECK(detail::is_pointwise(root.inputs()[0].primitive()));
    CHECK(array_equal(root, output).item<bool>());
  }

  auto square = reshape(x, {2, 3});
  auto output = transpose(square) + reshape(y, {3, 2});
  auto graph = optimize({x, y}, {output});
  CHECK_EQ(graph.outputs()[0].primitive().name(), std::string("Add"));
  CHECK(array_equal(graph.outputs()[0], output).item<bool>());
}

TEST_CASE("test compile reorder reshape after matmul exposes epilogue") {
  auto x = reshape(arange(6, float32), {2, 3});
  auto y = reshape(arange(12, float32), {3, 4});
  auto bias = full({2, 4}, array(1.0f));
  auto output = abs(reshape(matmul(x, y) + bias, {4, 2}));
  auto graph = optimize({x, y, bias}, {output}, false, CompileMode::enabled);
  const auto& root = graph.outputs()[0];
  CHECK(dynamic_cast<Reshape*>(&root.primitive()));
  CHECK(dynamic_cast<Compiled*>(&root.inputs()[0].primitive()));
  CHECK(array_equal(root, output).item<bool>());
}

TEST_CASE("test compile reorder selections preserves index operands") {
  auto x = reshape(arange(12, float32), {3, 4});
  auto y = negative(x);
  auto indices = array({2, 0, 2, 1, 0, 2, 1, 0}, {2, 4});
  const std::function<array(const array&)> selections[] = {
      [](const array& a) { return slice(a, {0, 0}, {3, 4}, {2, 2}); },
      [](const array& a) { return slice(a, {2, 3}, {-4, -5}, {-1, -2}); },
      [&](const array& a) { return take(a, array({-1, 0}), 0); },
      [&](const array& a) { return take_along_axis(a, indices, 0); }};
  for (const auto& selection : selections) {
    for (const auto& pointwise :
         {negative(x), add(x, y), where(greater(x, array(4)), x, y)}) {
      auto output = selection(pointwise);
      auto graph = optimize({x, y, indices}, {output});
      auto node = graph.outputs()[0];
      while (dynamic_cast<Reshape*>(&node.primitive()) ||
             dynamic_cast<Squeeze*>(&node.primitive())) {
        node = node.inputs()[0];
      }
      INFO(
          std::string(output.primitive().name()),
          " ",
          std::string(pointwise.primitive().name()),
          " -> ",
          std::string(node.primitive().name()));
      CHECK(detail::is_pointwise(node.primitive()));
      CHECK(array_equal(graph.outputs()[0], output).item<bool>());
    }
  }
}

TEST_CASE("test compile reorder concatenations preserves dtype and scalars") {
  auto x = array({-1.0f, 2.0f});
  auto y = array({3.0f, -4.0f, 5.0f});
  auto scalar = array(2.0f);
  auto condition = array(true);
  const std::function<array(const array&)> pointwise[] = {
      [](const array& a) { return negative(a); },
      [&](const array& a) { return a + scalar; },
      [&](const array& a) { return where(condition, a, scalar); }};
  for (const auto& op : pointwise) {
    auto output = concatenate({op(x), op(y)});
    auto graph = optimize({x, y, scalar, condition}, {output});
    CHECK(detail::is_pointwise(graph.outputs()[0].primitive()));
    CHECK(array_equal(graph.outputs()[0], output).item<bool>());
  }
  auto output = concatenate({astype(x, float16), astype(y, float16)});
  auto graph = optimize({x, y}, {output});
  CHECK_EQ(graph.outputs()[0].primitive().name(), std::string("Concatenate"));
  CHECK_EQ(graph.outputs()[0].dtype(), float16);
  CHECK(array_equal(graph.outputs()[0], output).item<bool>());

  auto a = array({1, 2}, int16);
  auto b = array({3, 4}, int32);
  output = concatenate({astype(a, float32), astype(b, float32)});
  graph = optimize({a, b}, {output});
  CHECK_EQ(graph.outputs()[0].primitive().name(), std::string("Concatenate"));
  CHECK(array_equal(graph.outputs()[0], output).item<bool>());
}

TEST_CASE("test compile reorder guards shared nodes streams and shapes") {
  auto x = reshape(arange(6, float32), {2, 3});
  auto view = transpose(x);
  auto pointwise = negative(x);
  auto stream = new_stream(default_device());
  std::vector<std::vector<array>> outputs{
      {abs(view), view},
      {take(pointwise, array({0}), 0), pointwise},
      {abs(transpose(x, stream))},
      {take(pointwise, array({0, 1, 0, 1}), 0)},
      {concatenate({pointwise, pointwise})},
      {concatenate({pointwise, abs(x)})}};
  for (const auto& roots : outputs) {
    auto graph = optimize({x}, roots);
    CHECK_EQ(
        graph.outputs()[0].primitive().name(),
        std::string(roots[0].primitive().name()));
    for (size_t i = 0; i < roots.size(); ++i) {
      CHECK(array_equal(graph.outputs()[i], roots[i]).item<bool>());
    }
  }
  detail::CompileContext context(
      true, detail::CompileTarget::Compile, CompileMode::no_fuse);
  CHECK_EQ(context.size_at_most(x, x), detail::ShapeCheck::Unknown);
  context.shapeless = false;
  CHECK_EQ(context.size_at_most(x, x), detail::ShapeCheck::True);
  CHECK_EQ(context.size_at_most(x, array(1)), detail::ShapeCheck::False);
}
