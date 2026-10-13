// Copyright © 2023-2024 Apple Inc.

// Required for using M_SQRT2 in MSVC.
#define _USE_MATH_DEFINES

#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "mlx/compile_impl.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

TEST_CASE("test compile graph isolation and cleanup") {
  auto x = array({-1.0f, 2.0f});
  auto inner = copy(x);
  auto outer = copy(inner);
  detail::CompileGraph graph({x}, {outer, outer, x}, {});
  auto root = graph.outputs()[0];
  auto input = root.inputs()[0];
  CHECK_NE(root.id(), outer.id());
  CHECK_NE(input.id(), inner.id());

  graph.replace_outputs({input}, {x});
  graph.validate();
  CHECK_EQ(root.inputs()[0].id(), x.id());
  CHECK_EQ(outer.inputs()[0].id(), inner.id());
  CHECK_FALSE(graph.contains(input));

  graph.replace_outputs({root}, {x});
  graph.validate();
  CHECK_EQ(graph.tape().size(), 1);
  CHECK_EQ(graph.tape()[0].id(), x.id());
  for (const auto& out : graph.outputs()) {
    CHECK_EQ(out.id(), x.id());
  }
  CHECK_EQ(graph.rewrite_count(), 2);
}

TEST_CASE("test compile graph simultaneous outputs") {
  auto x = array({9.0f, 7.0f});
  auto y = array({4.0f, 3.0f});
  auto parts = divmod(x, y);
  detail::CompileGraph graph(
      {x, y}, {parts[1], parts[0], parts[1], parts[0] + parts[1]}, {});
  auto remainder = graph.outputs()[0];
  auto quotient = graph.outputs()[1];
  auto sum = graph.outputs()[3];
  graph.replace_outputs({quotient, remainder}, {remainder, quotient});
  graph.validate();
  CHECK_EQ(graph.outputs()[0].id(), quotient.id());
  CHECK_EQ(graph.outputs()[1].id(), remainder.id());
  CHECK_EQ(graph.outputs()[2].id(), quotient.id());
  CHECK_EQ(sum.inputs()[0].id(), remainder.id());
  CHECK_EQ(sum.inputs()[1].id(), quotient.id());

  graph.replace_outputs({quotient}, {x});
  graph.validate();
  CHECK_EQ(graph.outputs()[0].id(), x.id());
  CHECK_EQ(graph.outputs()[1].id(), remainder.id());
  CHECK_EQ(sum.inputs()[0].id(), remainder.id());
  CHECK_EQ(sum.inputs()[1].id(), x.id());
  CHECK_EQ(remainder.siblings()[0].id(), quotient.id());
}

TEST_CASE("test compile graph retained dependencies") {
  auto x = array({-1.0f, 2.0f});
  auto a = abs(x);
  auto b = negative(a);
  detail::CompileGraph graph({x}, {a, b, b}, {});
  a = graph.outputs()[0];
  b = graph.outputs()[1];
  auto replacement = negative(a);
  graph.replace_outputs({a}, {replacement});
  graph.validate();
  CHECK_EQ(replacement.inputs()[0].id(), a.id());
  CHECK_EQ(b.inputs()[0].id(), replacement.id());
  CHECK_EQ(graph.outputs()[1].id(), graph.outputs()[2].id());
  CHECK(array_equal(b, array({1.0f, 2.0f})).item<bool>());
}

TEST_CASE("test compile graph validation") {
  auto x = array({-1.0f, 2.0f});
  detail::CompileGraph graph({x}, {negative(abs(x))}, {});
  auto root = graph.outputs()[0];
  auto input = root.inputs()[0];
  root.inputs()[0] = x;
  CHECK_THROWS_AS(graph.validate(), std::logic_error);
  root.inputs()[0] = root;
  CHECK_THROWS_AS(graph.validate(), std::logic_error);
  root.inputs()[0] = input;
  graph.validate();
}

TEST_CASE("test compile graph preserves fused tape") {
  auto x = array({-1.0f, 2.0f});
  auto a = abs(x);
  auto b = negative(a);
  detail::CompileGraph graph({x}, {b, a, b}, {});
  b = graph.outputs()[0];
  a = graph.outputs()[1];
  detail::compile_optimize(
      graph,
      detail::CompileContext(
          false, detail::CompileTarget::Compile, CompileMode::no_simplify));
  graph.validate();
  CHECK_EQ(b.inputs()[0].id(), a.id());
  CHECK_EQ(graph.outputs()[0].id(), graph.outputs()[2].id());
  const auto& primitive = graph.outputs()[0].primitive();
  CHECK_EQ(typeid(primitive), typeid(Compiled));
  CHECK_EQ(graph.outputs()[0].siblings()[0].id(), graph.outputs()[1].id());
  CHECK(array_equal(graph.outputs()[0], array({-1.0f, -2.0f})).item<bool>());
  CHECK(array_equal(graph.outputs()[1], array({1.0f, 2.0f})).item<bool>());
}

TEST_CASE("test compile graph invalid replacements") {
  auto x = array({1.0f, 2.0f});
  detail::CompileGraph graph({x}, {negative(x)}, {});
  auto a = graph.outputs()[0];
  CHECK_THROWS_AS(graph.replace_outputs({a}, {}), std::invalid_argument);
  CHECK_THROWS_AS(
      graph.replace_outputs({a}, {array(1.0f)}), std::invalid_argument);
  CHECK_THROWS_AS(
      graph.replace_outputs({a}, {astype(x, int32)}), std::invalid_argument);
  CHECK_THROWS_AS(graph.replace_outputs({a, a}, {x, a}), std::invalid_argument);
  CHECK_EQ(graph.outputs()[0].id(), a.id());
  CHECK_EQ(graph.rewrite_count(), 0);
  graph.validate();
}

TEST_CASE("test compile graph unused replacement region") {
  auto x = array({-1.0f, 2.0f});
  detail::CompileGraph graph({x}, {negative(abs(x))}, {});
  auto b = graph.outputs()[0];
  auto a = b.inputs()[0];
  auto unused = negative(a);
  auto retained = negative(b);
  graph.replace_outputs({a, b}, {unused, retained});
  CHECK_FALSE(graph.contains(unused));
  CHECK(graph.contains(a));
  CHECK(graph.contains(b));
  CHECK_EQ(b.inputs()[0].id(), a.id());
  graph.validate();
  CHECK(array_equal(graph.outputs()[0], array({1.0f, 2.0f})).item<bool>());
}

TEST_CASE("test compile graph shared broadcast") {
  auto x = array({-2.0f});
  auto fun = [](const std::vector<array>& inputs) -> std::vector<array> {
    auto b = broadcast_to(inputs[0], {4});
    return {negative(abs(b)), sum(b), b};
  };
  auto outputs = compile(fun)({x});
  const auto& primitive = outputs[0].primitive();
  CHECK_EQ(typeid(primitive), typeid(Compiled));
  auto expected = fun({x});
  for (size_t i = 0; i < outputs.size(); ++i) {
    CHECK(array_equal(outputs[i], expected[i]).item<bool>());
  }
}

TEST_CASE("test compile shape queries") {
  auto x = array(Shape{2, 1}, float32, nullptr, {});
  auto y = array(Shape{2, 1}, float32, nullptr, {});
  auto z = array(Shape{3, 1}, float32, nullptr, {});
  detail::CompileContext context(
      false, detail::CompileTarget::Compile, CompileMode::enabled);
  CHECK_EQ(context.same_shape(x, y), detail::ShapeCheck::True);
  CHECK_EQ(context.same_shape(x, z), detail::ShapeCheck::False);
  CHECK_EQ(context.dimension_is(x, 1, 1), detail::ShapeCheck::True);
  CHECK_EQ(context.dimension_is(x, 0, 1), detail::ShapeCheck::False);
  context.shapeless = true;
  CHECK_EQ(context.same_shape(x, y), detail::ShapeCheck::Unknown);
  CHECK_EQ(context.same_shape(x, z), detail::ShapeCheck::Unknown);
  CHECK_EQ(context.same_shape(x, x), detail::ShapeCheck::True);
  CHECK_EQ(context.same_shape(x, array(1)), detail::ShapeCheck::False);
  CHECK_EQ(context.dimension_is(x, 1, 1), detail::ShapeCheck::Unknown);
  CHECK_EQ(context.same_shape(sin(x), sin(x)), detail::ShapeCheck::True);
  CHECK_EQ(context.same_shape(sin(x), sin(y)), detail::ShapeCheck::Unknown);
  CHECK_EQ(context.same_shape(sin(x), cos(x)), detail::ShapeCheck::Unknown);
  CHECK_EQ(
      context.same_shape(broadcast_to(x, {2, 2}), broadcast_to(x, {2, 3})),
      detail::ShapeCheck::Unknown);
}

TEST_CASE("test compile pipeline export scope and streams") {
  auto x = array({1.0f, 2.0f});
  auto a = abs(x);
  detail::CompileGraph graph({x}, {copy(negative(a))}, {});
  detail::compile_optimize(
      graph,
      detail::CompileContext(
          false, detail::CompileTarget::Export, CompileMode::no_simplify));
  graph.validate();
  CHECK_EQ(graph.outputs()[0].primitive().name(), std::string("Negative"));
  CHECK_EQ(
      graph.outputs()[0].inputs()[0].primitive().name(), std::string("Abs"));

  auto stream = new_stream(default_device());
  auto b = abs(x, stream);
  detail::CompileGraph streams({x}, {a + b}, {});
  detail::compile_optimize(
      streams,
      detail::CompileContext(
          false, detail::CompileTarget::Compile, CompileMode::no_fuse));
  const auto& inputs = streams.outputs()[0].inputs();
  CHECK_NE(inputs[0].id(), inputs[1].id());
  CHECK_NE(inputs[0].primitive().stream(), inputs[1].primitive().stream());
  CHECK(array_equal(streams.outputs()[0], array({2.0f, 4.0f})).item<bool>());
}

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

TEST_CASE("test compile runner with a separate pass list") {
  auto remove_copy = [](detail::CompileGraph& graph,
                        const detail::CompileContext& context) {
    CHECK_EQ(context.log.stage, "canonicalize");
    if (context.log.pass == "compile_only") {
      CHECK_EQ(context.target, detail::CompileTarget::Compile);
    } else {
      CHECK_EQ(context.log.pass, "shared");
    }
    auto output = graph.outputs()[0];
    auto input = output.inputs()[0];
    graph.replace_outputs({output}, {input});
    context.log_rewrite("remove_copy", output, input);
  };
  const detail::PassSpec passes[] = {
      {"compile_only", remove_copy, detail::PassScope::CompileOnly},
      {"shared", remove_copy, detail::PassScope::CompileAndExport}};
  auto x = array({-1.0f, 2.0f});
  for (auto target :
       {detail::CompileTarget::Compile, detail::CompileTarget::Export}) {
    auto output = copy(copy(x));
    detail::CompileGraph graph({x}, {output, output}, {});
    detail::run_compile_passes(
        graph,
        detail::CompileContext(false, target, CompileMode::enabled),
        "canonicalize",
        passes);
    CHECK_EQ(graph.outputs()[0].id(), graph.outputs()[1].id());
    CHECK_EQ(
        graph.tape().size(), target == detail::CompileTarget::Compile ? 1 : 2);
    graph.validate();
    CHECK(array_equal(graph.outputs()[0], x).item<bool>());
  }
}

TEST_CASE("test disabled compile logs do not build messages") {
  detail::CompileLog log;
  log.categories = 0;
  bool built = false;
  log.write(detail::CompileLogCategory::Codegen, [&] {
    built = true;
    return "unused";
  });
  CHECK_FALSE(built);
}

std::vector<array> simple_fun(const std::vector<array>& inputs) {
  return std::vector<array>{inputs[0] + inputs[1]};
}

TEST_CASE("test simple compile") {
  auto compfn = compile(simple_fun);
  auto out = compfn({array(1.0f), array(2.0f)})[0];
  CHECK_EQ(out.item<float>(), 3.0f);

  out = compfn({array(1.0f), array(2.0f)})[0];
  CHECK_EQ(out.item<float>(), 3.0f);

  // Change the shapes
  out = compfn({array({1.0f, 2.0f}), array(2.0f)})[0];
  CHECK(array_equal(out, array({3.0f, 4.0f})).item<bool>());

  out = compfn({array(2.0f), array({1.0f, 2.0f})})[0];
  CHECK(array_equal(out, array({3.0f, 4.0f})).item<bool>());

  // Change the types
  out = compfn({array(2, int32), array({1.0f, 2.0f})})[0];
  CHECK(array_equal(out, array({3.0f, 4.0f})).item<bool>());

  out = compfn({array(2.0f), array({1, 2}, int32)})[0];
  CHECK(array_equal(out, array({3.0f, 4.0f})).item<bool>());
}

std::vector<array> grad_fun(const std::vector<array>& inputs) {
  auto loss = [](std::vector<array> ins) { return exp(ins[0] + ins[1]); };
  return grad(loss, {0, 1})(inputs);
}

TEST_CASE("test compile with grad") {
  auto x = array(1.0f);
  auto y = array(1.0f);
  auto grads_expected = grad_fun({x, y});
  auto grads_compile = compile(grad_fun)({x, y});
  CHECK(allclose(grads_compile[0], grads_expected[0]).item<bool>());
  CHECK(allclose(grads_compile[1], grads_expected[1]).item<bool>());
}

TEST_CASE("test compile inputs with primitive") {
  auto [k1, k2] = random::split(random::key(0));
  auto x = random::uniform({5, 5}, k1);
  auto y = random::uniform({5, 5}, k2);
  auto expected = simple_fun({x, y})[0];

  x = random::uniform({5, 5}, k1);
  y = random::uniform({5, 5}, k2);
  auto out = compile(simple_fun)({x, y})[0];
  CHECK(array_equal(expected, out).item<bool>());

  // Same thing twice
  out = compile(simple_fun)({x, y})[0];
  CHECK(array_equal(expected, out).item<bool>());
}

std::vector<array> fun_creats_array(const std::vector<array>& inputs) {
  return {inputs[0] + array(1.0f)};
}

TEST_CASE("test compile with created array") {
  auto cfun = compile(fun_creats_array);
  auto out = cfun({array(2.0f)});
  CHECK_EQ(out[0].item<float>(), 3.0f);

  // Try again
  out = cfun({array(2.0f)});
  CHECK_EQ(out[0].item<float>(), 3.0f);
}

std::vector<array> inner_fun(const std::vector<array>& inputs) {
  return {array(2) * inputs[0]};
}

std::vector<array> outer_fun(const std::vector<array>& inputs) {
  auto x = inputs[0] + inputs[1];
  auto y = compile(inner_fun)({x})[0];
  return {x + y};
}

TEST_CASE("test nested compile") {
  auto cfun = compile(outer_fun);
  auto out = cfun({array(1), array(2)})[0];
  CHECK_EQ(out.item<int>(), 9);

  // Try again
  out = cfun({array(1), array(2)})[0];
  CHECK_EQ(out.item<int>(), 9);
}

TEST_CASE("test enable and disable compile") {
  CHECK_THROWS(compile(nullptr));
  disable_compile();
  compile(nullptr);
  enable_compile();
  CHECK_THROWS(compile(nullptr));
}

std::vector<array> nan_const_fun(const std::vector<array>& inputs) {
  auto nan = array(std::numeric_limits<float>::quiet_NaN());
  return {where(greater(inputs[0], array(0.0f)), inputs[0], nan)};
}

std::vector<array> neg_inf_const_fun(const std::vector<array>& inputs) {
  auto inf = array(-std::numeric_limits<float>::infinity());
  return {where(greater(inputs[0], array(0.0f)), inputs[0], inf)};
}

TEST_CASE("test compile with non-finite constants") {
  // Regression test: baking a non-finite scalar constant (NaN / infinity) into
  // a fused compiled kernel used to stream a bare token (e.g. `nan`) into the
  // generated kernel source, which is not a valid identifier and broke
  // compilation (notably on the Metal backend).
  auto out = compile(nan_const_fun)({array({1.0f, -1.0f})})[0];
  eval(out);
  CHECK_EQ(out.shape(), Shape{2});
  CHECK_EQ(out.data<float>()[0], 1.0f);
  CHECK(std::isnan(out.data<float>()[1]));

  out = compile(neg_inf_const_fun)({array({1.0f, -1.0f})})[0];
  eval(out);
  CHECK_EQ(out.data<float>()[0], 1.0f);
  CHECK(std::isinf(out.data<float>()[1]));
  CHECK_LT(out.data<float>()[1], 0.0f);
}

auto add_scalars(const std::vector<array>&) {
  auto a = array(-1.0f);
  auto b = array(-1.0f);
  return std::vector<array>{abs(a), abs(b)};
};

auto max_scalars(const std::vector<array>&) {
  auto a = array({-1.0f, 2.0f});
  auto b = maximum(a, array(0.0f));
  auto c = maximum(-a, array(0.0f));
  auto d = b + c;
  return std::vector<array>{b, c, d};
};

TEST_CASE("test simplify scalars") {
  set_compile_mode(CompileMode::no_fuse);
  {
    auto cfun = compile(add_scalars);
    auto out = cfun({});
    auto c = out[0];
    auto d = out[1];
    CHECK(c.inputs()[0].id() == d.inputs()[0].id());
  }

  {
    auto a = array({-1.0f, 2.0f});
    auto out = compile(max_scalars)({a});
    auto b = out[0];
    auto c = out[1];
    auto d = out[2];
    CHECK(b.inputs()[1].id() == c.inputs()[1].id());
  }
  set_compile_mode(CompileMode::enabled);
}

auto exp_two(const std::vector<array>& inputs) {
  auto a = inputs[0];
  return std::vector<array>{exp(a) + exp(a)};
};

TEST_CASE("test simplify") {
  set_compile_mode(CompileMode::no_fuse);
  auto a = array({1.0f, 2.0f});
  auto b = compile(exp_two)({a})[0];
  CHECK(b.inputs()[0].id() == b.inputs()[1].id());
  set_compile_mode(CompileMode::enabled);
}

TEST_CASE("test simplify noops") {
  set_compile_mode(CompileMode::no_fuse);
  auto a = array({1.0f, 2.0f});
  auto fun = [](const std::vector<array>& inputs) -> std::vector<array> {
    return {copy(stop_gradient(exp(stop_gradient(inputs[0]))))};
  };
  auto b = compile(fun)({a})[0];
  CHECK(b.inputs()[0].id() == a.id());
  set_compile_mode(CompileMode::enabled);
}

auto add_diff(const std::vector<array>& inputs) {
  auto a = inputs[0];
  return std::vector<array>{cos(a) + sin(a)};
};

TEST_CASE("test no simplify") {
  set_compile_mode(CompileMode::no_fuse);
  auto a = array({1.0f, 2.0f});
  auto b = compile(add_diff)({a})[0];
  CHECK(b.inputs()[0].id() != b.inputs()[1].id());
  set_compile_mode(CompileMode::enabled);
}

auto multi_one(const std::vector<array>&) {
  auto a = array(1.0);
  auto b = array(2.0);
  auto c = divmod(a, b);
  auto d = divmod(a, b);
  auto e = c[0] + d[0];
  auto f = c[1] + d[1];
  return std::vector<array>{e, f};
}

auto multi_two(const std::vector<array>&) {
  auto a = array(1.0);
  auto b = array(1.0);
  return divmod(a, b);
}

auto multi_three(const std::vector<array>&) {
  auto a = array(1.0);
  auto b = array(2.0);
  auto c = divmod(a, b);
  auto d = divmod(a, b);
  auto e = stack({c[0], c[1], d[0], d[1]});
  return std::vector<array>{e};
}

TEST_CASE("test simplify multi output") {
  set_compile_mode(CompileMode::no_fuse);
  {
    auto out = compile(multi_one)({});
    auto e = out[0];
    auto f = out[1];
    CHECK_EQ(e.inputs()[0].id(), e.inputs()[1].id());
    CHECK_EQ(f.inputs()[0].id(), f.inputs()[1].id());
  }

  {
    auto c = compile(multi_two)({});
    CHECK_EQ(c[0].inputs()[0].id(), c[0].inputs()[1].id());
    CHECK_EQ(c[0].inputs()[0].id(), c[1].inputs()[0].id());
    CHECK_EQ(c[1].inputs()[0].id(), c[1].inputs()[1].id());
  }

  // Make sure the output order of multi-output primitives
  // is respected in simplification
  {
    auto e = compile(multi_three)({})[0];
    CHECK_EQ(e.inputs().size(), 4);
    CHECK_EQ(e.inputs().at(0).id(), e.inputs().at(2).id());
    CHECK_EQ(e.inputs().at(1).id(), e.inputs().at(3).id());
    CHECK(array_equal(e, array({0.0f, 1.0f, 0.0f, 1.0f})).item<bool>());
  }
  set_compile_mode(CompileMode::enabled);
}

// No fusion
auto unary_fused_0(const std::vector<array>& inputs) {
  return std::vector<array>{exp(inputs[0])};
}

// All compilable
auto unary_fused_1(const std::vector<array>& inputs) {
  return std::vector<array>{abs(negative(exp(inputs[0])))};
}

auto unary_fused_1_copy(const std::vector<array>& inputs) {
  return std::vector<array>{abs(negative(exp(inputs[0])))};
}

auto unary_fused_1_diff(const std::vector<array>& inputs) {
  return std::vector<array>{abs(exp(negative(inputs[0])))};
}

// Output into un-compilable primitive
auto unary_fused_2(const std::vector<array>& inputs) {
  return std::vector<array>{sum(abs(negative(exp(inputs[0]))), true)};
}

// Input from un-compilable primitive
auto unary_fused_3(const std::vector<array>& inputs) {
  return std::vector<array>{exp(abs(negative(sum(inputs[0], true))))};
}

TEST_CASE("test compile unary fused") {
  // NB: some of these tests are brittle and may need to be
  // updated if we change compile conditions
  {
    auto cfun = compile(unary_fused_0);
    auto x = array(2.0);
    auto out = cfun({x})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Exp));
    CHECK_EQ(out.inputs()[0].id(), x.id());
  }

  {
    auto cfun = compile(unary_fused_1);
    auto x = array(2.0);
    auto out = cfun({x})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    CHECK_EQ(out.inputs()[0].id(), x.id());

    auto expected_out = unary_fused_1({array(2.0)})[0];
    CHECK(allclose(out, expected_out).item<bool>());
  }

  {
    auto cfun = compile(unary_fused_2);
    auto x = array({1.0, 2.0});
    auto out = cfun({x});
    CHECK_EQ(out.size(), 1);

    auto& p = out[0].primitive();
    // NB: this test is brittle, will need to update
    // it if we change compile conditions
    CHECK_EQ(typeid(p), typeid(Reduce));
    auto cout = out[0].inputs()[0];
    auto& cp = cout.primitive();
    CHECK_EQ(typeid(cp), typeid(Compiled));
    CHECK_EQ(cout.inputs()[0].id(), x.id());
  }

  {
    auto cfun = compile(unary_fused_3);
    auto x = array({1.0, 2.0});
    auto out = cfun({x});

    auto& p = out[0].primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    auto sout = out[0].inputs()[0];
    CHECK_EQ(out[0].inputs().size(), 1);
    auto& sp = sout.primitive();
    CHECK_EQ(typeid(sp), typeid(Reduce));
    CHECK_EQ(sout.inputs()[0].id(), x.id());
  }

  // Is equivalent works
  {
    auto out1 = compile(unary_fused_1)({array(1.0)});
    auto out2 = compile(unary_fused_1_copy)({array(1.0)});
    CHECK(out1[0].primitive().is_equivalent(out2[0].primitive()));
    auto out3 = compile(unary_fused_1_diff)({array(1.0)});
    CHECK(!out1[0].primitive().is_equivalent(out3[0].primitive()));
  }
}

// All compilable
auto binary_fused_0(const std::vector<array>& inputs) {
  return std::vector<array>{inputs[0] + inputs[1]};
}

// Binary into unary
auto binary_fused_1(const std::vector<array>& inputs) {
  return std::vector<array>{abs(inputs[0] + inputs[1])};
}

// Binary into binary
auto binary_fused_2(const std::vector<array>& inputs) {
  auto x = inputs[0] + inputs[1];
  return std::vector<array>{x + inputs[0]};
}

// Binary into unary into un-compilable
auto binary_fused_3(const std::vector<array>& inputs) {
  return std::vector<array>{sum(abs(inputs[0] + inputs[1]), true)};
}

TEST_CASE("test compile binary fused") {
  {
    auto cfun = compile(binary_fused_0);
    auto x = array(2.0);
    auto y = array(2.0);
    auto out = cfun({x, y})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Add));
    CHECK_EQ(out.inputs()[0].id(), x.id());
  }

  {
    auto cfun = compile(binary_fused_1);
    auto x = array(2.0);
    auto y = array(2.0);
    auto out = cfun({x, y})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    CHECK_EQ(out.inputs()[0].id(), x.id());
    CHECK_EQ(out.inputs()[1].id(), y.id());

    auto expected_out = binary_fused_1({x, y})[0];
    CHECK_EQ(out.item<float>(), expected_out.item<float>());
  }

  {
    auto cfun = compile(binary_fused_2);
    auto x = array(2.0);
    auto y = array(2.0);
    auto out = cfun({x, y})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    CHECK_EQ(out.inputs()[0].id(), x.id());
    CHECK_EQ(out.inputs()[1].id(), y.id());
  }

  {
    auto cfun = compile(binary_fused_3);
    auto x = array({1.0, 2.0});
    auto y = array({1.0, 2.0});
    auto out = cfun({x, y})[0];

    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Reduce));

    auto cout = out.inputs()[0];
    auto& cp = cout.primitive();
    CHECK_EQ(typeid(cp), typeid(Compiled));
    CHECK_EQ(cout.inputs()[0].id(), x.id());
    CHECK_EQ(cout.inputs()[1].id(), y.id());
  }
}

auto gelu_1(const std::vector<array>& inputs) {
  auto& x = inputs[0];
  auto out = x * (1.0f + erf(x / M_SQRT2)) / 2.0f;
  return std::vector<array>{out};
}

TEST_CASE("test compile gelu") {
  {
    auto cfun = compile(gelu_1);
    auto x = array(1.0);
    auto out = cfun({x})[0];
    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    CHECK_EQ(out.inputs().size(), 4);
    for (auto& in : out.inputs()) {
      CHECK(in.inputs().empty());
    }
    auto expected_out = gelu_1({x})[0];
    CHECK(allclose(out, expected_out).item<bool>());
  }

  {
    auto cfun = compile(gelu_1);
    auto x = array({1.0, 0.5});
    auto out = cfun({x})[0];
    auto& p = out.primitive();
    CHECK_EQ(typeid(p), typeid(Compiled));
    CHECK_EQ(out.inputs().size(), 4);
    for (auto& in : out.inputs()) {
      CHECK(in.inputs().empty());
    }

    auto expected_out = gelu_1({x})[0];
    CHECK(allclose(out, expected_out).item<bool>());
  }
}

// Uncompilable input outside fused tape
auto unary_with_two_outputs(const std::vector<array>& inputs) {
  auto x = exp(inputs[0]);
  return std::vector<array>{exp(x), sum(x, true)};
}

auto uncompilable_inputs(const std::vector<array>& inputs) {
  auto& x = inputs[0];
  auto& y = inputs[1];
  return std::vector<array>{x * abs(exp(y)), sum(x, true)};
}

auto uncompilable_inputs_order_matters(const std::vector<array>& inputs) {
  auto& x = inputs[0];
  auto& y = inputs[1];
  return std::vector<array>{x / abs(exp(y)), sum(x, true)};
}

TEST_CASE("test compile tape with outside consumers") {
  {
    auto cfun = compile(unary_with_two_outputs);
    auto x = array({2.0, 2.0});
    auto out = cfun({x});

    auto& p1 = out[0].primitive();
    CHECK_EQ(typeid(p1), typeid(Exp));
    auto& p2 = out[1].primitive();
    CHECK_EQ(typeid(p2), typeid(Reduce));
  }

  {
    auto cfun = compile(uncompilable_inputs);
    auto x = array({2.0, 2.0});
    auto y = array({1.6, 0.6});
    auto outs = cfun({x, y});

    auto& p1 = outs[0].primitive();
    CHECK_EQ(typeid(p1), typeid(Compiled));
    auto& p2 = outs[1].primitive();
    CHECK_EQ(typeid(p2), typeid(Reduce));
    CHECK_EQ(outs[0].inputs().size(), 2);

    auto expected_outs = uncompilable_inputs({x, y});
    CHECK(allclose(outs[0], expected_outs[0]).item<bool>());
    CHECK(allclose(outs[1], expected_outs[1]).item<bool>());
  }

  {
    auto cfun = compile(uncompilable_inputs_order_matters);
    auto x = array({2.0, 2.0});
    auto y = array({1.6, 0.6});
    auto outs = cfun({x, y});

    auto& p1 = outs[0].primitive();
    CHECK_EQ(typeid(p1), typeid(Compiled));
    auto& p2 = outs[1].primitive();
    CHECK_EQ(typeid(p2), typeid(Reduce));
    CHECK_EQ(outs[0].inputs().size(), 2);

    auto expected_outs = uncompilable_inputs_order_matters({x, y});
    CHECK(allclose(outs[0], expected_outs[0]).item<bool>());
    CHECK(allclose(outs[1], expected_outs[1]).item<bool>());
  }
}

auto compile_across_streams(const std::vector<array>& inputs) {
  auto s2 = new_stream(default_device());
  auto x = exp(abs(inputs[0]));
  auto y = exp(abs(x, s2), s2);
  return std::vector<array>{y};
}

TEST_CASE("test compile across streams") {
  auto cfun = compile(compile_across_streams);
  auto x = array({2.0f});
  auto out = cfun({x})[0];
  auto& p1 = out.primitive();
  CHECK_EQ(typeid(p1), typeid(Compiled));
  CHECK_EQ(out.inputs().size(), 1);
  auto input = out.inputs()[0];
  auto& p2 = input.primitive();
  CHECK_EQ(typeid(p2), typeid(Compiled));
  CHECK_EQ(input.inputs()[0].id(), x.id());
}

auto unary_compile_outputs(const std::vector<array>& inputs) {
  auto x = abs(inputs[0]);
  auto y = square(x);
  return std::vector<array>{x, y};
}

auto binary_compile_outputs(const std::vector<array>& inputs) {
  auto x = inputs[0];
  auto y = inputs[1];
  x = x + y;
  y = x + y;
  return std::vector<array>{x, y};
}

TEST_CASE("test compile internal output") {
  {
    auto cfun = compile(unary_compile_outputs);
    auto x = array({3, -2});
    auto outs = cfun({x});
    auto& p1 = outs[0].primitive();
    CHECK_EQ(typeid(p1), typeid(Compiled));
    auto& p2 = outs[1].primitive();
    CHECK_EQ(typeid(p2), typeid(Compiled));
    CHECK_EQ(outs[0].siblings()[0].id(), outs[1].id());
    auto expected_outs = unary_compile_outputs({x});
    CHECK(array_equal(outs[0], expected_outs[0]).item<bool>());
    CHECK(array_equal(outs[1], expected_outs[1]).item<bool>());
  }

  {
    auto cfun = compile(binary_compile_outputs);
    auto x = array({3, -2});
    auto y = array({1, -1});
    auto outs = cfun({x, y});
    auto& p1 = outs[0].primitive();
    CHECK_EQ(typeid(p1), typeid(Compiled));
    auto& p2 = outs[1].primitive();
    CHECK_EQ(typeid(p2), typeid(Compiled));
    auto expected_outs = binary_compile_outputs({x, y});
    CHECK(array_equal(outs[0], expected_outs[0]).item<bool>());
    CHECK(array_equal(outs[1], expected_outs[1]).item<bool>());
  }
}

auto deep_unary_compile(const std::vector<array>& inputs) {
  auto x = inputs[0];
  for (int i = 0; i < 10; ++i) {
    x = cos(sin(x));
  }
  return std::vector<array>{x};
}

TEST_CASE("test compile deep graph") {
  auto cfun = compile(deep_unary_compile);
  auto x = array({3.0f, -2.0f});
  auto out = cfun({x})[0];
  auto expected_out = deep_unary_compile({x})[0];
  CHECK(allclose(out, expected_out).item<bool>());
}

auto repeat_input_to_compiled(const std::vector<array>& inputs) {
  auto x = abs(exp(inputs[0]));
  auto y = abs(exp(sum(x)));
  return std::vector<array>{x + y};
}

TEST_CASE("test compile repeat input") {
  auto cfun = compile(repeat_input_to_compiled);
  auto x = array({3.0f, -2.0f});
  auto out = cfun({x})[0];
  auto expected_out = repeat_input_to_compiled({x})[0];
  CHECK(allclose(out, expected_out).item<bool>());
}

auto compile_unary_inner(const std::vector<array>& inputs) {
  auto x = inputs[0];
  return std::vector<array>{exp(exp(x))};
}

auto compile_unary_outer(const std::vector<array>& inputs) {
  auto cfun = compile(compile_unary_inner);
  return cfun(cfun(inputs));
}

TEST_CASE("test compile compiled function") {
  auto cfun = compile(compile_unary_outer);
  auto x = array({1.0f});
  auto out = cfun({x})[0];
  auto& p = out.primitive();
  CHECK_EQ(typeid(p), typeid(Compiled));
  CHECK_EQ(out.inputs()[0].id(), x.id());
}

auto grad_unary_compiled(const std::vector<array>& inputs) {
  auto gradfn = value_and_grad(compile(compile_unary_inner));
  auto [out, grad] = gradfn(inputs);
  return std::vector{out[0], grad[0]};
}

TEST_CASE("test transform compiled function") {
  auto cfun = compile(grad_unary_compiled);
  auto x = array(1.0f);
  auto outs = cfun({x});
  auto& p = outs[0].primitive();
  CHECK_EQ(typeid(p), typeid(Compiled));
  CHECK_EQ(outs[0].siblings()[0].id(), outs[1].id());
  CHECK(!outs[0].inputs()[0].has_primitive());
  CHECK(!outs[0].inputs()[1].has_primitive());
}

TEST_CASE("test fusion kernel reuse") {
  auto cfun = compile(gelu_1);
  auto x = array({2.0f, -2.0f});
  auto y = cfun({x})[0];
  auto p = std::dynamic_pointer_cast<Compiled>(y.primitive_ptr());
  eval(y);

  std::string lib_name = p->lib_name();
  CHECK(!lib_name.empty());

  x = astype(reshape(arange(10), {2, 5}), float32);
  auto z = cfun({x})[0];
  auto pz = std::dynamic_pointer_cast<Compiled>(z.primitive_ptr());
  eval(z);

  std::string lib_name_z = pz->lib_name();
  CHECK(!lib_name_z.empty());

  CHECK_EQ(lib_name, lib_name_z);
}

auto add3(const std::vector<array>& xs) {
  return std::vector<array>{xs[0] + xs[0] + xs[0]};
}

TEST_CASE("test fusion types") {
  auto cfun = compile(add3);
  auto x = array({2.0f, -2.0f});
  auto y = cfun({x})[0];
  auto p = std::dynamic_pointer_cast<Compiled>(y.primitive_ptr());
  eval(y);

  std::string lib_name = p->lib_name();
  CHECK(!lib_name.empty());

  x = array({2, -2}, int32);
  auto z = cfun({x})[0];
  auto pz = std::dynamic_pointer_cast<Compiled>(z.primitive_ptr());
  eval(z);

  std::string lib_name_z = pz->lib_name();
  CHECK(!lib_name_z.empty());
}

auto compile_shapeless_not_ok(const std::vector<array>& inputs) {
  auto x = reshape(inputs[0], {2, 2});
  return std::vector<array>{x};
}

auto compile_shapeless_ok(const std::vector<array>& inputs) {
  auto x = inputs[0] + array({2});
  return std::vector<array>{x};
}

TEST_CASE("test shapeless compile") {
  {
    auto cfun = compile(compile_shapeless_not_ok, /* shapeless */ true);
    cfun({array({1, 2, 3, 4})});
    CHECK_THROWS(cfun({array({1, 2, 3, 4, 5})}));
  }

  {
    auto cfun = compile(compile_shapeless_ok, /* shapeless */ true);
    auto out = cfun({array({1, 2})})[0];
    auto out2 = cfun({array({1, 2, 3, 4})})[0];

    // Not making a new constant array since no recompile,
    // hence the ids should be the same
    CHECK_EQ(out.inputs()[1].id(), out2.inputs()[1].id());
    CHECK(array_equal(out2, array({3, 4, 5, 6})).item<bool>());

    // Recompile since type changes
    out2 = cfun({array({1.0, 2.0})})[0];
    CHECK_NE(out.inputs()[1].id(), out2.inputs()[1].id());

    // Recompile since ndim changes
    out2 = cfun({array({1.0, 2.0}, {1, 2})})[0];
    CHECK_NE(out.inputs()[1].id(), out2.inputs()[1].id());
  }
}

auto compile_broadcast_add(const std::vector<array>& inputs) {
  auto b = zeros({8, 8});
  return std::vector<array>{inputs[0] + b};
}

TEST_CASE("test compile strides") {
  {
    auto cfun = compile(compile_broadcast_add);
    auto a = zeros({1, 8, 8});
    auto out = cfun({a})[0];
    eval(out);
    CHECK_EQ(out.strides().size(), 3);
  }
}

TEST_CASE("test compile change streams") {
  auto cfun = compile(simple_fun);
  auto out = cfun({array(1.0f), array(2.0f)})[0];
  CHECK_EQ(out.primitive().stream(), default_stream(default_device()));

  auto s = new_stream(default_device());
  StreamContext sctx(s);
  out = cfun({array(1.0f), array(2.0f)})[0];
  CHECK_EQ(out.primitive().stream(), s);
}

TEST_CASE("test compile lambda") {
  auto fun = [](const std::vector<array>& inputs) {
    return std::vector<array>{abs(inputs[0])};
  };

  auto out = compile(fun)({array(-1)});
  CHECK_EQ(out[0].item<int>(), 1);

  decltype(compile(nullptr)) c_local_fun;
  {
    auto local_fun = [](const std::vector<array>& inputs) {
      return std::vector<array>{abs(inputs[0])};
    };
    c_local_fun = compile(local_fun);
  }

  // This is ok even though local_fun is out of scope
  out = c_local_fun({array(-1)});
  CHECK_EQ(out[0].item<int>(), 1);

  {
    int x = 2;
    auto local_fun = [x](const std::vector<array>& inputs) {
      return std::vector<array>{inputs[0] + x};
    };
    c_local_fun = compile(local_fun);
  }
  // Also ok even though local_fun is out of scope.
  out = c_local_fun({array(0)});
  CHECK_EQ(out[0].item<int>(), 2);

  int x = 2;
  auto fun_with_capture = [&x](const std::vector<array>& inputs) {
    return std::vector<array>{inputs[0] + x};
  };
  auto cfun = compile(fun_with_capture);
  out = cfun({array(0)});
  CHECK_EQ(out[0].item<int>(), 2);

  // Doesn't recompile
  x = 3;
  out = cfun({array(0)});
  CHECK_EQ(out[0].item<int>(), 2);

  // Recompiles
  auto cfun2 = compile(fun_with_capture);
  out = cfun2({array(0)});
  CHECK_EQ(out[0].item<int>(), 3);
}

TEST_CASE("test compile with no-ops") {
  auto fun = [](const std::vector<array>& inputs) {
    return std::vector<array>{abs(stop_gradient(abs(inputs[0])))};
  };
  auto in = array(1.0);
  auto out = compile(fun)({in})[0];
  CHECK_EQ(out.inputs()[0].id(), in.id());
}

TEST_CASE("test compile random bits") {
  auto fun = [](const std::vector<array>& inputs) {
    auto key = inputs[0];
    auto a = random::bits({32, 32}, 4, key);
    auto b = random::bits({32, 32}, 2, key);
    return std::vector<array>{a + b};
  };
  auto in = random::key(0);
  auto expected = fun({in})[0];
  auto out = compile(fun)({in})[0];
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test compile throwing first trace does not poison cache") {
  // A nullary function whose first trace raises. Without rolling the cache
  // entry back, the second call with matching (empty) inputs would hit a
  // half-filled entry, skip tracing, and silently return empty outputs.
  bool should_throw = true;
  auto fun = [&should_throw](const std::vector<array>& inputs) {
    if (should_throw) {
      throw std::runtime_error("trace failure");
    }
    return std::vector<array>{array(1.0f) + array(2.0f)};
  };

  auto cfun = compile(fun);

  // First call: the trace raises and must propagate.
  CHECK_THROWS_AS(cfun({}), std::runtime_error);

  // Second call: still raising. It must re-trace and raise again rather than
  // return an empty result from a poisoned cache entry.
  CHECK_THROWS_AS(cfun({}), std::runtime_error);

  // Once the function stops raising, a retry must trace cleanly and succeed.
  should_throw = false;
  auto out = cfun({});
  REQUIRE_EQ(out.size(), 1);
  CHECK_EQ(out[0].item<float>(), 3.0f);
}
