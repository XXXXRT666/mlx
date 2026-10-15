// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <map>

#include <fmt/format.h>

#include "mlx/compile_impl.h"
#include "mlx/primitives.h"

namespace mlx::core::detail {
namespace {

constexpr int max_compile_depth = 11;
constexpr int max_compile_arrays = 24;

bool is_unary(const Primitive& p) {
  return (
      typeid(p) == typeid(Abs) || typeid(p) == typeid(ArcCos) ||
      typeid(p) == typeid(ArcCosh) || typeid(p) == typeid(ArcSin) ||
      typeid(p) == typeid(ArcSinh) || typeid(p) == typeid(ArcTan) ||
      typeid(p) == typeid(ArcTanh) || typeid(p) == typeid(AsType) ||
      typeid(p) == typeid(Ceil) || typeid(p) == typeid(Cos) ||
      typeid(p) == typeid(Conjugate) || typeid(p) == typeid(Cosh) ||
      typeid(p) == typeid(Remainder) || typeid(p) == typeid(Erf) ||
      typeid(p) == typeid(ErfInv) || typeid(p) == typeid(Exp) ||
      typeid(p) == typeid(Floor) || typeid(p) == typeid(Log) ||
      typeid(p) == typeid(Log1p) || typeid(p) == typeid(LogicalNot) ||
      typeid(p) == typeid(Negative) || typeid(p) == typeid(Round) ||
      typeid(p) == typeid(Sigmoid) || typeid(p) == typeid(Sign) ||
      typeid(p) == typeid(Sin) || typeid(p) == typeid(Sinh) ||
      typeid(p) == typeid(Square) || typeid(p) == typeid(Sqrt) ||
      typeid(p) == typeid(Tan) || typeid(p) == typeid(Tanh) ||
      typeid(p) == typeid(Expm1) || typeid(p) == typeid(Real) ||
      typeid(p) == typeid(Imag) || typeid(p) == typeid(BitwiseInvert));
}

bool is_binary(const Primitive& p) {
  return typeid(p) == typeid(Add) || typeid(p) == typeid(ArcTan2) ||
      typeid(p) == typeid(BitwiseBinary) || typeid(p) == typeid(Divide) ||
      typeid(p) == typeid(Equal) || typeid(p) == typeid(FloorDivide) ||
      typeid(p) == typeid(Greater) || typeid(p) == typeid(GreaterEqual) ||
      typeid(p) == typeid(Less) || typeid(p) == typeid(LessEqual) ||
      typeid(p) == typeid(LogicalNot) || typeid(p) == typeid(LogicalAnd) ||
      typeid(p) == typeid(LogicalOr) || typeid(p) == typeid(LogAddExp) ||
      typeid(p) == typeid(Maximum) || typeid(p) == typeid(Minimum) ||
      typeid(p) == typeid(Multiply) || typeid(p) == typeid(NotEqual) ||
      typeid(p) == typeid(Power) || typeid(p) == typeid(Subtract);
}

bool is_ternary(const Primitive& p) {
  return typeid(p) == typeid(Select);
}

bool is_broadcast(const Primitive& p) {
  return typeid(p) == typeid(Broadcast);
}

bool is_fusable(const Primitive& p) {
  return is_pointwise(p) || is_broadcast(p);
}

std::unordered_set<uintptr_t> collect_fusable_nodes(
    CompileGraph& graph,
    const CompileContext& context,
    std::span<const array> roots) {
  std::function<void(const array&, int, const Stream&)> recurse;
  std::unordered_set<uintptr_t> cache;
  std::unordered_set<uintptr_t> input_set;
  recurse = [&](const array& a, int depth, const Stream& s) {
    if (cache.find(a.id()) != cache.end()) {
      return;
    }

    std::string_view reason;
    if (depth >= max_compile_depth) {
      reason = "depth_limit";
    } else if (!a.has_primitive()) {
      reason = "graph_input";
    } else if (a.primitive().stream() != s) {
      reason = "stream_mismatch";
    } else if (!is_fusable(a.primitive())) {
      reason = "unsupported_primitive";
    } else if (depth == 0 && is_broadcast(a.primitive())) {
      // A broadcast root can materialize a larger contiguous output.
      reason = "broadcast_output_materialization";
    } else if (graph.is_output(a)) {
      auto same_shape = context.same_shape(a, roots.back());
      if (same_shape != ShapeCheck::True) {
        reason = same_shape == ShapeCheck::False ? "output_shape_mismatch"
                                                 : "output_shape_not_proven";
      }
    }
    if (!reason.empty()) {
      context.log.write(CompileLogCategory::Fusion, [&] {
        return fmt::format(
            "rule=pointwise_region node={} reason={}", a.id(), reason);
      });
      input_set.insert(a.id());
      return;
    }

    bool all_consumers_in = true;
    if (depth > 0) {
      auto& consumers = graph.consumers().at(a.id());
      for (auto& [consumer, index] : consumers) {
        auto in_cache = cache.find(consumer.id()) != cache.end();
        if (!in_cache) {
          all_consumers_in = false;
          break;
        }
      }
    }

    // Split broadcasts that also have consumers outside the fused region.
    if (!all_consumers_in) {
      if (a.has_primitive() && is_broadcast(a.primitive()) &&
          input_set.size() < max_compile_arrays) {
        auto copy = graph.split_uses(a, cache);
        recurse(copy, depth, s);
      } else {
        context.log.write(CompileLogCategory::Fusion, [&] {
          return fmt::format(
              "rule=pointwise_region node={} reason={}",
              a.id(),
              "external_consumer");
        });
        input_set.insert(a.id());
      }
      return;
    }

    if (graph.is_output(a)) {
      input_set.insert(a.id());
    } else {
      // Not an input anymore since fusing it
      input_set.erase(a.id());
    }
    if (input_set.size() >= max_compile_arrays) {
      context.log.write(CompileLogCategory::Fusion, [&] {
        return fmt::format(
            "rule=pointwise_region node={} reason={}", a.id(), "array_limit");
      });
      return;
    }
    cache.insert({a.id()});

    for (auto& in : a.inputs()) {
      recurse(in, depth + 1, s);
    }
  };

  for (const auto& root : roots) {
    if (root.has_primitive()) {
      recurse(root, 0, root.primitive().stream());
    }
  }
  return cache;
}

void fuse_subgraph(
    CompileGraph& graph,
    const CompileContext& context,
    std::span<const array> roots,
    const std::unordered_set<uintptr_t>& cache,
    const std::unordered_set<uintptr_t>& input_ids) {
  // Collect inputs and order the selected nodes in a topological tape.
  std::unordered_set<uintptr_t> input_set;
  std::vector<array> inputs;
  std::vector<array> fused_tape;
  std::unordered_set<uintptr_t> tape_set;
  std::function<void(const array&)> recurse_tape;
  recurse_tape = [&](const array& a) {
    if (cache.find(a.id()) == cache.end()) {
      if (input_set.find(a.id()) == input_set.end()) {
        input_set.insert(a.id());
        inputs.push_back(a);
      }
      return;
    }
    if (tape_set.find(a.id()) != tape_set.end()) {
      return;
    }
    tape_set.insert(a.id());
    for (auto& in : a.inputs()) {
      recurse_tape(in);
    }
    fused_tape.push_back(a);
  };
  for (const auto& root : roots) {
    recurse_tape(root);
  }

  std::vector<array> old_outputs;
  for (const auto& node : fused_tape) {
    if (graph.is_output(node)) {
      old_outputs.push_back(node);
    }
  }
  for (const auto& root : roots) {
    if (!graph.is_output(root)) {
      old_outputs.push_back(root);
    }
  }

  std::vector<Shape> shapes;
  std::vector<Dtype> types;
  for (auto& o : old_outputs) {
    if (context.same_shape(o, old_outputs.back()) != ShapeCheck::True) {
      throw std::runtime_error(
          "[compile] Compilation failed. Tried to fuse operations with different output shapes");
    }
    shapes.push_back(o.shape());
    types.push_back(o.dtype());
  }
  std::unordered_set<uintptr_t> constant_ids;
  for (auto& in : inputs) {
    // Scalar constant
    if (in.size() == 1 && !in.has_primitive() &&
        input_ids.find(in.id()) == input_ids.end()) {
      constant_ids.insert(in.id());
    }
  }
  auto compiled_outputs = array::make_arrays(
      std::move(shapes),
      types,
      std::make_shared<Compiled>(
          old_outputs.back().primitive().stream(),
          inputs,
          old_outputs,
          std::move(fused_tape),
          std::move(constant_ids),
          context.log.compilation_id),
      inputs);

  graph.replace_outputs(old_outputs, compiled_outputs);
  auto rule =
      roots.size() > 1 ? "horizontal_pointwise_region" : "pointwise_region";
  if (context.log.enabled(CompileLogCategory::Passes)) {
    for (size_t i = 0; i < old_outputs.size(); ++i) {
      context.log_rewrite(rule, old_outputs[i], compiled_outputs[i]);
    }
  }
  context.log.write(CompileLogCategory::Fusion, [&] {
    return fmt::format(
        "rule={} node={} reason=accepted primitive={}",
        rule,
        roots.back().id(),
        compiled_outputs.back().primitive_id());
  });
}

} // namespace

bool is_pointwise(const Primitive& p) {
  return is_unary(p) || is_binary(p) || is_ternary(p);
}

void fuse_horizontal_pointwise(
    CompileGraph& graph,
    const CompileContext& context) {
  if (graph.outputs().size() < 2) {
    return;
  }
  std::unordered_set<uintptr_t> input_ids;
  for (const auto& input : graph.inputs()) {
    input_ids.insert(input.id());
  }
  std::unordered_set<uintptr_t> seen;
  std::map<std::pair<uintptr_t, Stream>, std::vector<array>> groups;
  for (const auto& output : graph.outputs()) {
    auto consumers = graph.consumers().find(output.id());
    if (!seen.insert(output.id()).second || !output.has_primitive() ||
        !is_pointwise(output.primitive()) ||
        (consumers != graph.consumers().end() && !consumers->second.empty())) {
      continue;
    }
    for (const auto& input : output.inputs()) {
      auto& group = groups[{input.id(), output.primitive().stream()}];
      if (group.empty() || group.back().id() != output.id()) {
        group.push_back(output);
      }
    }
  }
  for (const auto& [_, group] : groups) {
    for (size_t offset = 0; offset < group.size();
         offset += max_compile_arrays) {
      std::vector<array> roots;
      auto end = std::min(group.size(), offset + max_compile_arrays);
      for (size_t i = offset; i < end; ++i) {
        const auto& root = group[i];
        if (!graph.contains(root)) {
          continue;
        }
        auto shape = roots.empty() ? ShapeCheck::True
                                   : context.same_shape(root, roots[0]);
        if (shape != ShapeCheck::True) {
          context.log.write(CompileLogCategory::Fusion, [&] {
            return fmt::format(
                "rule=horizontal_pointwise_region node={} reason={}",
                root.id(),
                shape == ShapeCheck::False ? "output_shape_mismatch"
                                           : "output_shape_not_proven");
          });
          continue;
        }
        roots.push_back(root);
      }
      // Each fusion removes at least two roots from this bounded group.
      while (roots.size() > 1) {
        auto nodes = collect_fusable_nodes(graph, context, roots);
        auto selected = roots;
        std::erase_if(selected, [&](const auto& root) {
          return !nodes.contains(root.id());
        });
        if (selected.size() < 2) {
          break;
        }
        fuse_subgraph(graph, context, selected, nodes, input_ids);
        std::erase_if(
            roots, [&](const auto& root) { return !graph.contains(root); });
      }
    }
  }
}

void fuse_pointwise(CompileGraph& graph, const CompileContext& context) {
  std::unordered_set<uintptr_t> input_ids;
  for (const auto& in : graph.inputs()) {
    input_ids.insert(in.id());
  }

  auto candidates = graph.tape();
  while (!candidates.empty()) {
    auto root = candidates.back();
    candidates.pop_back();
    if (!graph.contains(root)) {
      continue;
    }
    std::span<const array> roots(&root, 1);
    auto nodes = collect_fusable_nodes(graph, context, roots);
    if (nodes.size() <= 1) {
      if (nodes.size() == 1) {
        context.log.write(CompileLogCategory::Fusion, [&] {
          return fmt::format(
              "rule=pointwise_region node={} reason={}",
              root.id(),
              "single_primitive");
        });
      }
      continue;
    }
    fuse_subgraph(graph, context, roots, nodes, input_ids);
  }
}

} // namespace mlx::core::detail
