// Copyright © 2026 Apple Inc.

#include <numeric>

#include "doctest/doctest.h"

#include "mlx/compile_impl.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

TEST_CASE("test compile simplify nested duplicates") {
  auto x = array({-1.0f, 2.0f});
  auto a = x;
  auto b = x;
  for (int i = 0; i < 8; ++i) {
    a = sin(a);
    b = sin(b);
  }
  detail::CompileGraph graph({x}, {a + b}, {});
  detail::compile_optimize(
      graph,
      detail::CompileContext(
          false, detail::CompileTarget::Compile, CompileMode::no_fuse));
  graph.validate();
  CHECK_EQ(graph.tape().size(), 10);
  const auto& inputs = graph.outputs()[0].inputs();
  CHECK_EQ(inputs[0].id(), inputs[1].id());
  CHECK(array_equal(graph.outputs()[0], a + b).item<bool>());
}

TEST_CASE("test compile pass shape guards") {
  auto x = array({-1.0f, 2.0f});
  auto y = array({3.0f, 4.0f});
  for (bool shapeless : {false, true}) {
    detail::CompileContext context(
        shapeless, detail::CompileTarget::Compile, CompileMode::no_fuse);
    detail::CompileGraph duplicates({x}, {sin(x) + sin(x)}, {});
    detail::compile_optimize(duplicates, context);
    duplicates.validate();
    const auto& inputs = duplicates.outputs()[0].inputs();
    CHECK_EQ(inputs[0].id(), inputs[1].id());

    auto a = abs(x);
    detail::CompileGraph outputs({x, y}, {a, a + y, a}, {});
    context.mode = CompileMode::enabled;
    detail::compile_optimize(outputs, context);
    outputs.validate();
    const auto& primitive = outputs.outputs()[0].primitive();
    CHECK_EQ(typeid(primitive) == typeid(Compiled), !shapeless);
    CHECK_EQ(outputs.outputs()[0].id(), outputs.outputs()[2].id());
  }
}

TEST_CASE("test compile simplification pass order") {
  auto x = array({-1.0f, 2.0f});
  for (bool shapeless : {false, true}) {
    for (auto target :
         {detail::CompileTarget::Compile, detail::CompileTarget::Export}) {
      auto a = sin(copy(x) + array(1.0f));
      auto b = sin(stop_gradient(copy(x)) + array(1.0f));
      auto output = a + b;
      detail::CompileGraph graph({x}, {output, output}, {});
      detail::compile_optimize(
          graph,
          detail::CompileContext(
              shapeless,
              target,
              target == detail::CompileTarget::Compile
                  ? CompileMode::no_fuse
                  : CompileMode::no_simplify));
      graph.validate();
      CHECK_EQ(graph.outputs()[0].id(), graph.outputs()[1].id());
      const auto& inputs = graph.outputs()[0].inputs();
      CHECK_EQ(inputs[0].id(), inputs[1].id());
      for (const auto& node : graph.tape()) {
        if (node.has_primitive()) {
          const auto& primitive = node.primitive();
          CHECK_NE(typeid(primitive), typeid(Copy));
          CHECK_NE(typeid(primitive), typeid(StopGradient));
          CHECK_NE(typeid(primitive), typeid(Compiled));
        }
      }
      CHECK(array_equal(graph.outputs()[0], output).item<bool>());
    }
  }
}

TEST_CASE("test compile inverse transposes") {
  for (const Shape& shape :
       {Shape{}, Shape{4}, Shape{2, 3, 4}, Shape{0, 3, 4}}) {
    auto x = array(shape, float32, nullptr, {});
    std::vector<int> axes(shape.size());
    std::iota(axes.begin(), axes.end(), 0);
    do {
      std::vector<int> inverse(axes.size());
      for (int i = 0; i < axes.size(); ++i) {
        inverse[axes[i]] = i;
      }
      for (bool shapeless : {false, true}) {
        for (auto target :
             {detail::CompileTarget::Compile, detail::CompileTarget::Export}) {
          auto output = transpose(transpose(x, axes), inverse);
          detail::CompileGraph graph({x}, {output, output}, {});
          detail::compile_optimize(
              graph,
              detail::CompileContext(shapeless, target, CompileMode::no_fuse));
          graph.validate();
          CHECK_EQ(graph.tape().size(), 1);
          CHECK_EQ(graph.outputs()[0].id(), x.id());
          CHECK_EQ(graph.outputs()[1].id(), x.id());
        }
      }
    } while (std::next_permutation(axes.begin(), axes.end()));
  }
}

TEST_CASE("test compile inverse transpose consumers and guards") {
  auto x = reshape(arange(9), {3, 3});
  auto inner = transpose(x);
  auto outer = transpose(inner);
  detail::CompileContext context(
      false, detail::CompileTarget::Compile, CompileMode::no_fuse);
  detail::CompileGraph shared({x}, {inner, outer, outer, negative(outer)}, {});
  detail::compile_optimize(shared, context);
  shared.validate();
  CHECK_EQ(shared.outputs()[0].primitive().name(), std::string("Transpose"));
  CHECK_EQ(shared.outputs()[1].id(), x.id());
  CHECK_EQ(shared.outputs()[2].id(), x.id());
  CHECK_EQ(shared.outputs()[3].inputs()[0].id(), x.id());

  auto stream = new_stream(default_device());
  for (auto output : {transpose(inner, {0, 1}), transpose(inner, stream)}) {
    detail::CompileGraph graph({x}, {output}, {});
    detail::compile_optimize(graph, context);
    graph.validate();
    CHECK_EQ(graph.outputs()[0].primitive().name(), std::string("Transpose"));
    CHECK(array_equal(graph.outputs()[0], output).item<bool>());
  }
  detail::CompileGraph disabled({x}, {outer}, {});
  context.mode = CompileMode::no_simplify;
  detail::compile_optimize(disabled, context);
  CHECK_EQ(disabled.outputs()[0].primitive().name(), std::string("Transpose"));

  auto chain = x;
  for (int i = 0; i < 8; ++i) {
    chain = transpose(chain);
  }
  detail::CompileGraph graph({x}, {chain}, {});
  context.mode = CompileMode::no_fuse;
  detail::compile_optimize(graph, context);
  graph.validate();
  CHECK_EQ(graph.outputs()[0].id(), x.id());
}

TEST_CASE("test compile roundtrip views") {
  for (const Shape& shape : {Shape{2, 3}, Shape{0, 3}, Shape{}}) {
    auto x = array(shape, float32, nullptr, {});
    auto middle_shape = shape.empty() ? Shape{1} : Shape{shape[0] * 3};
    auto inner = reshape(x, middle_shape);
    auto outer = reshape(inner, shape);
    for (bool shapeless : {false, true}) {
      for (auto target :
           {detail::CompileTarget::Compile, detail::CompileTarget::Export}) {
        detail::CompileGraph graph({x}, {inner, outer, outer}, {});
        detail::compile_optimize(
            graph,
            detail::CompileContext(shapeless, target, CompileMode::no_fuse));
        graph.validate();
        CHECK_EQ(graph.outputs()[0].primitive().name(), std::string("Reshape"));
        CHECK_EQ(
            graph.outputs()[1].id() == x.id(), !shapeless || shape.empty());
        CHECK_EQ(graph.outputs()[1].id(), graph.outputs()[2].id());
      }
    }
  }
}

TEST_CASE("test compile roundtrip view guards") {
  auto x = reshape(arange(6), {2, 3});
  auto inner = reshape(x, {3, 2});
  auto stream = new_stream(default_device());
  detail::CompileContext context(
      false, detail::CompileTarget::Compile, CompileMode::no_fuse);
  for (auto output :
       {reshape(inner, {6}),
        reshape(inner, x.shape(), stream),
        reshape(transpose(inner), {6})}) {
    detail::CompileGraph graph({x}, {output}, {});
    detail::compile_optimize(graph, context);
    graph.validate();
    CHECK_NE(graph.outputs()[0].id(), x.id());
    CHECK(array_equal(graph.outputs()[0], output).item<bool>());
  }
  auto chain = x;
  for (int i = 0; i < 8; ++i) {
    chain = reshape(reshape(chain, {6}), {2, 3});
  }
  detail::CompileGraph graph({x}, {chain}, {});
  detail::compile_optimize(graph, context);
  graph.validate();
  CHECK_EQ(graph.outputs()[0].id(), x.id());
}

TEST_CASE("test compile split concat elimination") {
  for (const Shape& shape : {Shape{2, 12}, Shape{0, 12}, Shape{2, 0}}) {
    auto x = array(shape, float32, nullptr, {});
    for (const Shape& indices :
         {Shape{3, 7},
          Shape{0, 3, 12},
          Shape{3, 3, 7},
          Shape{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}) {
      auto parts = split(x, indices, 1);
      auto output = concatenate(parts, 1);
      for (bool shapeless : {false, true}) {
        for (auto target :
             {detail::CompileTarget::Compile, detail::CompileTarget::Export}) {
          detail::CompileGraph graph({x}, {output, parts[0], output}, {});
          detail::compile_optimize(
              graph,
              detail::CompileContext(shapeless, target, CompileMode::no_fuse));
          graph.validate();
          CHECK_EQ(graph.outputs()[0].id() == x.id(), !shapeless);
          CHECK_EQ(graph.outputs()[0].id(), graph.outputs()[2].id());
          CHECK_EQ(graph.outputs()[1].shape(), parts[0].shape());
        }
      }
    }
  }
}

TEST_CASE("test compile split concat guards") {
  auto x = astype(reshape(arange(24), {2, 12}), int32);
  auto parts = split(x, {4, 8}, 1);
  auto other = split(x + array(1), {4, 8}, 1);
  std::vector<Shape> shapes;
  for (const auto& part : parts) {
    shapes.push_back(part.shape());
  }
  auto replayed = array::make_arrays(
      std::move(shapes),
      std::vector<Dtype>(parts.size(), x.dtype()),
      parts[0].primitive_ptr(),
      {x + array(1)});
  auto stream = new_stream(default_device());
  std::vector<array> outputs{
      concatenate({parts[2], parts[1], parts[0]}, 1),
      concatenate({parts[0], parts[0], parts[2]}, 1),
      concatenate({parts[0], negative(parts[1]), parts[2]}, 1),
      concatenate({parts[0], other[1], parts[2]}, 1),
      concatenate({parts[0], replayed[1], parts[2]}, 1),
      concatenate(parts, 0),
      concatenate(parts, 1, stream),
      concatenate({slice(x, {0, 0}, {2, 6}), slice(x, {0, 4}, {2, 10})}, 1),
      concatenate(
          {slice(x, {0, 0}, {2, 12}, {1, 2}),
           slice(x, {0, 1}, {2, 12}, {1, 2})},
          1)};
  for (const auto& output : outputs) {
    detail::CompileGraph graph({x}, {output}, {});
    detail::compile_optimize(
        graph,
        detail::CompileContext(
            false, detail::CompileTarget::Compile, CompileMode::no_fuse));
    graph.validate();
    CHECK_EQ(graph.outputs()[0].primitive().name(), std::string("Concatenate"));
    CHECK_EQ(graph.outputs()[0].dtype(), output.dtype());
    CHECK(array_equal(graph.outputs()[0], output).item<bool>());
  }

  auto fresh_parts = split(x, {4, 8}, 1);
  auto promoted = concatenate(
      {astype(fresh_parts[0], float32), fresh_parts[1], fresh_parts[2]}, 1);
  detail::CompileGraph graph({x}, {promoted}, {});
  detail::compile_optimize(
      graph,
      detail::CompileContext(
          false, detail::CompileTarget::Compile, CompileMode::no_fuse));
  graph.validate();
  const auto& output = graph.outputs()[0];
  CHECK_EQ(output.primitive().name(), std::string("AsType"));
  CHECK_EQ(output.inputs()[0].id(), x.id());
  CHECK_EQ(output.dtype(), float32);
  CHECK(array_equal(output, promoted).item<bool>());
}

TEST_CASE("test compile view elimination exposes pointwise fusion") {
  const std::function<array(const array&)> views[] = {
      [](const array& a) { return transpose(transpose(a)); },
      [](const array& a) { return reshape(reshape(a, {6}), {2, 3}); },
      [](const array& a) { return concatenate(split(a, {1, 2}, 1), 1); }};
  for (auto dtype : {float32, float16, bfloat16}) {
    auto x = astype(reshape(arange(6) * array(0.1f), {2, 3}), dtype);
    for (const auto& view : views) {
      auto fun = [&](const std::vector<array>& inputs) {
        return std::vector<array>{abs(view(exp(inputs[0])))};
      };
      auto output = compile(fun)({x})[0];
      const auto& primitive = output.primitive();
      CHECK_EQ(typeid(primitive), typeid(Compiled));
      CHECK_EQ(output.dtype(), dtype);
      CHECK(allclose(output, fun({x})[0], 1e-3, 1e-3).item<bool>());
    }
  }
}
