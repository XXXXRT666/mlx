// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <map>
#include <optional>

#include <fmt/format.h>

#include "mlx/compile_impl.h"
#include "mlx/primitives.h"
#include "mlx/transforms_impl.h"

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

bool is_noop(const Primitive& p) {
  return typeid(p) == typeid(Copy) || typeid(p) == typeid(StopGradient);
}

bool is_fusable(const Primitive& p) {
  return is_pointwise(p) || is_broadcast(p);
}

static inline uint64_t splitmix64(uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

struct VecU64Hash {
  size_t operator()(const std::vector<uint64_t>& s) const noexcept {
    uint64_t h =
        0x243f6a8885a308d3ull ^ (uint64_t)s.size() * 0x9e3779b97f4a7c15ull;
    for (uint64_t x : s) {
      h = splitmix64(x ^ splitmix64(h + 0x9e3779b97f4a7c15ull));
    }
    return (size_t)h;
  }
};

void deduplicate_scalars(CompileGraph& graph, const CompileContext& context) {
  std::map<std::pair<uint64_t, Dtype::Val>, array> scalars;
  auto is_scalar = [](const array& a) {
    return a.is_available() && a.ndim() == 0;
  };
  auto get_scalar_rep = [](const array& a) {
    uint64_t v = 0;
    switch (a.dtype().size()) {
      case 1:
        v = *a.data<uint8_t>();
        break;
      case 2:
        v = *a.data<uint16_t>();
        break;
      case 4:
        v = *a.data<uint32_t>();
        break;
      case 8:
        v = *a.data<uint64_t>();
        break;
    }
    return std::make_pair(v, a.dtype().val());
  };

  std::vector<array> old_scalars;
  std::vector<array> new_scalars;
  for (const auto& a : graph.tape()) {
    if (is_scalar(a)) {
      auto [it, inserted] = scalars.emplace(get_scalar_rep(a), a);
      if (!inserted && !graph.is_output(a)) {
        old_scalars.push_back(a);
        new_scalars.push_back(it->second);
      }
    }
  }
  graph.replace_outputs(old_scalars, new_scalars);
  if (context.log.enabled(CompileLogCategory::Passes)) {
    for (size_t i = 0; i < old_scalars.size(); ++i) {
      context.log_rewrite("same_scalar", old_scalars[i], new_scalars[i]);
    }
  }
}

void remove_noops(CompileGraph& graph, const CompileContext& context) {
  // Worklists hold candidates only; graph queries are refreshed after rewrites.
  auto candidates = graph.tape();
  std::reverse(candidates.begin(), candidates.end());
  while (!candidates.empty()) {
    auto a = candidates.back();
    candidates.pop_back();
    if (graph.contains(a) && a.has_primitive() && is_noop(a.primitive())) {
      graph.replace_outputs({a}, {a.inputs()[0]});
      context.log_rewrite("remove_noop", a, a.inputs()[0]);
    }
  }
}

void eliminate_inverse_transposes(
    CompileGraph& graph,
    const CompileContext& context) {
  auto candidates = graph.tape();
  for (const auto& a : candidates) {
    if (!graph.contains(a) || !a.has_primitive()) {
      continue;
    }
    auto& primitive = a.primitive();
    if (typeid(primitive) != typeid(Transpose)) {
      continue;
    }
    const auto& inner = a.inputs()[0];
    if (!inner.has_primitive()) {
      continue;
    }
    auto& inner_primitive = inner.primitive();
    if (typeid(inner_primitive) != typeid(Transpose)) {
      continue;
    }
    auto p = static_cast<const Transpose&>(inner_primitive).state();
    auto q = static_cast<const Transpose&>(primitive).state();
    std::string_view reason;
    if (primitive.stream() != inner_primitive.stream()) {
      reason = "stream_mismatch";
    } else {
      for (int i = 0; i < q.size(); ++i) {
        if (p[q[i]] != i) {
          reason = "index_map_mismatch";
          break;
        }
      }
    }
    context.log.write(CompileLogCategory::Guards, [&] {
      return fmt::format(
          "rule=inverse_transposes check_stage=compile check=inverse_permutation node={} result={} reason={}",
          a.id(),
          reason.empty() ? "True" : "False",
          reason.empty() ? "proven" : reason);
    });
    if (reason.empty()) {
      auto input = inner.inputs()[0];
      graph.replace_outputs({a}, {input});
      context.log_rewrite("inverse_transposes", a, input);
    }
  }
}

void eliminate_roundtrip_views(
    CompileGraph& graph,
    const CompileContext& context) {
  auto candidates = graph.tape();
  for (const auto& a : candidates) {
    if (!graph.contains(a) || !a.has_primitive()) {
      continue;
    }
    auto& primitive = a.primitive();
    if (typeid(primitive) != typeid(Reshape)) {
      continue;
    }
    const auto& inner = a.inputs()[0];
    if (!inner.has_primitive()) {
      continue;
    }
    auto& inner_primitive = inner.primitive();
    if (typeid(inner_primitive) != typeid(Reshape)) {
      continue;
    }
    auto input = inner.inputs()[0];
    std::string_view reason;
    if (primitive.stream() != inner_primitive.stream()) {
      reason = "stream_mismatch";
    } else {
      auto shape = context.same_shape(a, input);
      if (shape != ShapeCheck::True) {
        reason =
            shape == ShapeCheck::False ? "shape_mismatch" : "shape_not_proven";
      }
    }
    context.log.write(CompileLogCategory::Guards, [&] {
      return fmt::format(
          "rule=roundtrip_views check_stage=compile node={} reason={}",
          a.id(),
          reason.empty() ? "proven" : reason);
    });
    if (reason.empty()) {
      graph.replace_outputs({a}, {input});
      context.log_rewrite("roundtrip_views", a, input);
    }
  }
}

void eliminate_split_concat(
    CompileGraph& graph,
    const CompileContext& context) {
  auto candidates = graph.tape();
  for (const auto& a : candidates) {
    if (!graph.contains(a) || !a.has_primitive()) {
      continue;
    }
    auto& primitive = a.primitive();
    if (typeid(primitive) != typeid(Concatenate)) {
      continue;
    }
    auto axis = static_cast<const Concatenate&>(primitive).state();
    const auto& parts = a.inputs();
    const auto& first = parts[0];
    auto* split = dynamic_cast<Split*>(first.primitive_ptr().get());
    auto* first_slice = dynamic_cast<Slice*>(first.primitive_ptr().get());
    auto input = split || first_slice ? first.inputs()[0] : first;
    auto shape = context.same_shape(a, input);
    std::string_view reason;
    if (shape != ShapeCheck::True) {
      reason =
          shape == ShapeCheck::False ? "shape_mismatch" : "shape_not_proven";
    } else if (split) {
      auto [indices, split_axis] = split->state();
      if (split_axis != axis || parts.size() != indices.size() + 1) {
        reason = "index_map_mismatch";
      } else if (split->stream() != primitive.stream()) {
        reason = "stream_mismatch";
      } else {
        auto outputs = first.outputs();
        for (size_t i = 0; i < parts.size(); ++i) {
          if (parts[i].id() != outputs[i].id()) {
            reason = "index_map_mismatch";
            break;
          }
        }
      }
    } else {
      int offset = 0;
      for (const auto& part : parts) {
        if (part.id() == input.id()) {
          if (offset != 0) {
            reason = "index_map_mismatch";
            break;
          }
          offset = input.shape(axis);
          continue;
        }
        auto* slice = dynamic_cast<Slice*>(part.primitive_ptr().get());
        if (!slice || part.inputs()[0].id() != input.id()) {
          reason = "index_map_mismatch";
          break;
        }
        if (slice->stream() != primitive.stream()) {
          reason = "stream_mismatch";
          break;
        }
        auto [start, stop, strides] = slice->state();
        for (int d = 0; d < input.ndim(); ++d) {
          if (strides[d] != 1 || start[d] != (d == axis ? offset : 0) ||
              (d != axis && stop[d] != input.shape(d))) {
            reason = "index_map_mismatch";
            break;
          }
        }
        if (stop[axis] < offset || stop[axis] > input.shape(axis)) {
          reason = "index_map_mismatch";
        }
        if (!reason.empty()) {
          break;
        }
        offset = stop[axis];
      }
      if (reason.empty() && offset != input.shape(axis)) {
        reason = "index_map_mismatch";
      }
    }
    context.log.write(CompileLogCategory::Guards, [&] {
      return fmt::format(
          "rule=split_concat check_stage=compile node={} reason={}",
          a.id(),
          reason.empty() ? "proven" : reason);
    });
    if (reason.empty()) {
      graph.replace_outputs({a}, {input});
      context.log_rewrite("split_concat", a, input);
    }
  }
}

void eliminate_common_subexpressions(
    CompileGraph& graph,
    const CompileContext& context) {
  auto array_equivalent = [&](const array& a, const array& b) {
    if (a.primitive_id() == b.primitive_id() ||
        a.primitive().stream() != b.primitive().stream() ||
        a.inputs().size() != b.inputs().size()) {
      return false;
    }
    for (size_t i = 0; i < a.inputs().size(); ++i) {
      if (a.inputs()[i].id() != b.inputs()[i].id()) {
        return false;
      }
    }
    auto a_outputs = a.outputs();
    auto b_outputs = b.outputs();
    if (a_outputs.size() != b_outputs.size()) {
      return false;
    }
    for (size_t i = 0; i < a_outputs.size(); ++i) {
      if (context.same_shape(a_outputs[i], b_outputs[i]) != ShapeCheck::True ||
          a_outputs[i].dtype() != b_outputs[i].dtype()) {
        return false;
      }
    }
    const auto& pa = a.primitive();
    const auto& pb = b.primitive();
    return typeid(pa) == typeid(pb) && pa.is_equivalent(pb);
  };

  // Inputs have their final representatives before each node is visited.
  std::unordered_map<std::vector<uint64_t>, std::vector<array>, VecU64Hash>
      equivalent;
  auto candidates = graph.tape();
  std::reverse(candidates.begin(), candidates.end());
  while (!candidates.empty()) {
    auto a = candidates.back();
    candidates.pop_back();
    if (!graph.contains(a) || !a.has_primitive() || a.inputs().empty()) {
      continue;
    }
    std::vector<uint64_t> key;
    for (const auto& in : a.inputs()) {
      key.push_back(in.id());
    }
    const auto& primitive = a.primitive();
    key.push_back(typeid(primitive).hash_code());
    auto& group = equivalent[key];
    auto outputs = a.outputs();
    bool is_output = std::any_of(outputs.begin(), outputs.end(), [&](auto& o) {
      return graph.is_output(o);
    });
    auto match = group.end();
    if (!is_output) {
      match = std::find_if(group.begin(), group.end(), [&](const auto& b) {
        return graph.contains(b) && array_equivalent(a, b);
      });
    }
    if (match == group.end()) {
      group.push_back(a);
    } else {
      auto replacements = match->outputs();
      graph.replace_outputs(outputs, replacements);
      if (context.log.enabled(CompileLogCategory::Passes)) {
        for (size_t i = 0; i < outputs.size(); ++i) {
          context.log_rewrite(
              "equivalent_primitive", outputs[i], replacements[i]);
        }
      }
    }
  }
}

std::unordered_set<uintptr_t> collect_fusable_nodes(
    CompileGraph& graph,
    const CompileContext& context,
    const array& root) {
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
      auto same_shape = context.same_shape(a, root);
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

  if (root.has_primitive()) {
    Stream s = root.primitive().stream();
    recurse(root, 0, s);
  }
  return cache;
}

void fuse_subgraph(
    CompileGraph& graph,
    const CompileContext& context,
    const array& root,
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
  recurse_tape(root);

  std::vector<array> old_outputs;
  for (size_t j = 0; j + 1 < fused_tape.size(); ++j) {
    if (graph.is_output(fused_tape[j])) {
      old_outputs.push_back(fused_tape[j]);
    }
  }
  old_outputs.push_back(root);

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
  if (context.log.enabled(CompileLogCategory::Passes)) {
    for (size_t i = 0; i < old_outputs.size(); ++i) {
      context.log_rewrite(
          "pointwise_region", old_outputs[i], compiled_outputs[i]);
    }
  }
  context.log.write(CompileLogCategory::Fusion, [&] {
    return fmt::format(
        "rule=pointwise_region node={} reason=accepted primitive={}",
        root.id(),
        compiled_outputs.back().primitive_id());
  });
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
    auto nodes = collect_fusable_nodes(graph, context, root);
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
    fuse_subgraph(graph, context, root, nodes, input_ids);
  }
}

} // namespace

bool is_pointwise(const Primitive& p) {
  return is_unary(p) || is_binary(p) || is_ternary(p);
}

void run_compile_passes(
    CompileGraph& graph,
    const CompileContext& context,
    std::string_view stage,
    std::span<const PassSpec> passes) {
  InTracing tracing(context.shapeless);
  std::optional<InExportTracing> export_tracing;
  if (context.target == CompileTarget::Export) {
    export_tracing.emplace();
  }
  for (const auto& pass : passes) {
    if (context.target == CompileTarget::Export &&
        pass.scope == PassScope::CompileOnly) {
      continue;
    }
    auto scoped_context = context;
    scoped_context.log.stage = stage;
    scoped_context.log.pass = pass.name;
    auto before = graph.rewrite_count();
    auto nodes_before = graph.tape().size();
    pass.run(graph, scoped_context);
#ifndef NDEBUG
    graph.validate();
#else
    graph.rebuild();
#endif
    scoped_context.log.write(CompileLogCategory::Passes, [&] {
      return fmt::format(
          "event=complete target={} rewrites={} nodes={}->{}",
          context.target == CompileTarget::Compile ? "compile" : "export",
          graph.rewrite_count() - before,
          nodes_before,
          graph.tape().size());
    });
  }
}

void compile_optimize(CompileGraph& graph, const CompileContext& context) {
  static constexpr PassSpec canonicalize_passes[] = {
      {"deduplicate_scalars", deduplicate_scalars, PassScope::CompileAndExport},
      {"remove_noops", remove_noops, PassScope::CompileAndExport},
      {"reorder_pointwise_views",
       reorder_pointwise_views,
       PassScope::CompileAndExport},
      {"reorder_pointwise_selections",
       reorder_pointwise_selections,
       PassScope::CompileAndExport},
      {"reorder_pointwise_concatenations",
       reorder_pointwise_concatenations,
       PassScope::CompileAndExport},
      {"eliminate_inverse_transposes",
       eliminate_inverse_transposes,
       PassScope::CompileAndExport},
      {"eliminate_roundtrip_views",
       eliminate_roundtrip_views,
       PassScope::CompileAndExport},
      {"eliminate_split_concat",
       eliminate_split_concat,
       PassScope::CompileAndExport},
      {"eliminate_common_subexpressions",
       eliminate_common_subexpressions,
       PassScope::CompileAndExport},
  };
  static constexpr PassSpec fusion_passes[] = {
      {"fuse_pointwise", fuse_pointwise, PassScope::CompileOnly},
  };
  if (context.target == CompileTarget::Export ||
      context.mode != CompileMode::no_simplify) {
    run_compile_passes(graph, context, "canonicalize", canonicalize_passes);
  }
  if (context.mode != CompileMode::no_fuse) {
    run_compile_passes(graph, context, "fusion", fusion_passes);
  }
}

} // namespace mlx::core::detail
