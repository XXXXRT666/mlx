// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <map>
#include <optional>

#include <fmt/format.h>

#include "mlx/compile_impl.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core::detail {
namespace {

bool is_noop(const Primitive& p) {
  return typeid(p) == typeid(Copy) || typeid(p) == typeid(StopGradient);
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

bool lossless_integer_cast(Dtype from, Dtype to) {
  if (from == bool_) {
    return to == bool_ || issubdtype(to, integer);
  }
  if (!issubdtype(from, integer) || !issubdtype(to, integer)) {
    return false;
  }
  bool from_signed = issubdtype(from, signedinteger);
  bool to_signed = issubdtype(to, signedinteger);
  return from_signed == to_signed ? from.size() <= to.size()
                                  : !from_signed && from.size() < to.size();
}

bool preserves_uniform_value(const Primitive& p) {
  return typeid(p) == typeid(Full) || typeid(p) == typeid(Broadcast) ||
      typeid(p) == typeid(BroadcastAxes) || typeid(p) == typeid(Reshape) ||
      typeid(p) == typeid(Transpose) || typeid(p) == typeid(Flatten) ||
      typeid(p) == typeid(Unflatten) || typeid(p) == typeid(Squeeze) ||
      typeid(p) == typeid(ExpandDims);
}

std::optional<bool> fold_boolean(
    const Primitive& p,
    const std::vector<std::optional<bool>>& inputs) {
  if (!inputs[0]) {
    return std::nullopt;
  }
  bool a = *inputs[0];
  if (typeid(p) == typeid(LogicalNot)) {
    return !a;
  }
  if (inputs.size() != 2 || !inputs[1]) {
    return std::nullopt;
  }
  bool b = *inputs[1];
  if (typeid(p) == typeid(LogicalAnd)) {
    return a && b;
  }
  if (typeid(p) == typeid(LogicalOr)) {
    return a || b;
  }
  if (typeid(p) == typeid(Equal)) {
    return a == b;
  }
  if (typeid(p) == typeid(NotEqual)) {
    return a != b;
  }
  return std::nullopt;
}

bool is_identity_constant(
    array scalar,
    uint64_t value,
    const std::unordered_set<uintptr_t>& input_ids,
    Stream stream) {
  while (scalar.has_primitive()) {
    if (input_ids.contains(scalar.id()) ||
        scalar.primitive().stream() != stream) {
      return false;
    }
    auto& p = scalar.primitive();
    auto integral = [](Dtype dtype) {
      return dtype == bool_ || issubdtype(dtype, integer);
    };
    bool broadcast = typeid(p) == typeid(Broadcast) &&
        scalar.inputs().size() == 1 && scalar.inputs()[0].ndim() == 0;
    // Zero and one are exact through integer and bool casts.
    bool cast = typeid(p) == typeid(AsType) && scalar.ndim() == 0 &&
        integral(scalar.dtype()) && integral(scalar.inputs()[0].dtype());
    if (!broadcast && !cast) {
      return false;
    }
    scalar = scalar.inputs()[0];
  }
  if (scalar.ndim() != 0 || !scalar.is_available() ||
      input_ids.contains(scalar.id()) ||
      !(scalar.dtype() == bool_ || issubdtype(scalar.dtype(), integer))) {
    return false;
  }
  switch (scalar.itemsize()) {
    case 1:
      return *scalar.data<uint8_t>() == value;
    case 2:
      return *scalar.data<uint16_t>() == value;
    case 4:
      return *scalar.data<uint32_t>() == value;
    case 8:
      return *scalar.data<uint64_t>() == value;
  }
  return false;
}

void log_skip(
    const CompileContext& context,
    const array& node,
    std::string_view reason) {
  context.log.write(CompileLogCategory::Guards, [&] {
    return fmt::format(
        "rule={} check_stage=compile node={} reason={}",
        context.log.pass,
        node.id(),
        reason);
  });
}

} // namespace

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

void eliminate_concat_split(
    CompileGraph& graph,
    const CompileContext& context) {
  auto candidates = graph.tape();
  for (const auto& concat : candidates) {
    if (!graph.contains(concat) || !concat.has_primitive()) {
      continue;
    }
    auto* primitive = dynamic_cast<Concatenate*>(&concat.primitive());
    auto uses = graph.consumers().find(concat.id());
    if (!primitive || uses == graph.consumers().end()) {
      continue;
    }
    auto consumers = uses->second;
    if (std::none_of(consumers.begin(), consumers.end(), [](const auto& use) {
          return dynamic_cast<Split*>(&use.first.primitive()) ||
              dynamic_cast<Slice*>(&use.first.primitive());
        })) {
      continue;
    }
    auto axis = primitive->state();
    auto inputs = concat.inputs();
    Shape boundaries;
    std::map<std::pair<int, int>, array> segments;
    int offset = 0;
    for (const auto& in : inputs) {
      if (context.dimension_is(in, axis, in.shape(axis)) != ShapeCheck::True) {
        break;
      }
      int end = offset + in.shape(axis);
      boundaries.push_back(end);
      segments.emplace(std::make_pair(offset, end), in);
      offset = end;
    }
    if (boundaries.size() != inputs.size()) {
      log_skip(context, concat, "segment_boundaries_not_proven");
      continue;
    }
    boundaries.pop_back();
    for (const auto& [node, _] : consumers) {
      if (!graph.contains(node)) {
        continue;
      }
      auto* split = dynamic_cast<Split*>(&node.primitive());
      auto* slice = dynamic_cast<Slice*>(&node.primitive());
      if (!split && !slice) {
        continue;
      }
      if (node.primitive().stream() != primitive->stream()) {
        log_skip(context, node, "stream_mismatch");
        continue;
      }
      auto outputs = node.outputs();
      std::vector<array> replacements;
      if (split) {
        auto [indices, split_axis] = split->state();
        if (split_axis == axis && indices == boundaries) {
          replacements = inputs;
        }
      } else {
        auto [start, stop, strides] = slice->state();
        bool full = true;
        for (int d = 0; d < concat.ndim(); ++d) {
          full &= strides[d] == 1 &&
              (d == axis || (start[d] == 0 && stop[d] == concat.shape(d)));
        }
        auto segment = segments.find({start[axis], stop[axis]});
        if (full && segment != segments.end()) {
          replacements.push_back(segment->second);
        }
      }
      if (replacements.size() != outputs.size() ||
          !std::equal(
              outputs.begin(),
              outputs.end(),
              replacements.begin(),
              [&](const auto& out, const auto& in) {
                return context.same_shape(out, in) == ShapeCheck::True;
              })) {
        log_skip(context, node, "segment_mapping_not_proven");
        continue;
      }
      graph.replace_outputs(outputs, replacements);
      for (size_t i = 0; i < outputs.size(); ++i) {
        context.log_rewrite("concat_split", outputs[i], replacements[i]);
      }
    }
  }
}

void eliminate_redundant_casts(
    CompileGraph& graph,
    const CompileContext& context) {
  auto candidates = graph.tape();
  for (const auto& node : candidates) {
    if (!graph.contains(node) ||
        !dynamic_cast<AsType*>(node.primitive_ptr().get())) {
      continue;
    }
    auto inner = node.inputs()[0];
    if (!dynamic_cast<AsType*>(inner.primitive_ptr().get())) {
      continue;
    }
    if (node.primitive().stream() != inner.primitive().stream()) {
      log_skip(context, node, "stream_mismatch");
      continue;
    }
    auto input = inner.inputs()[0];
    if (!lossless_integer_cast(input.dtype(), inner.dtype()) ||
        !(node.dtype() == bool_ || issubdtype(node.dtype(), integer))) {
      log_skip(context, node, "lossy_or_unsupported_conversion");
      continue;
    }
    auto output = node.dtype() == input.dtype()
        ? input
        : array(node.shape(), node.dtype(), node.primitive_ptr(), {input});
    graph.replace_outputs({node}, {output});
    context.log_rewrite("redundant_cast", node, output);
  }
}

void combine_concatenate_slices(
    CompileGraph& graph,
    const CompileContext& context) {
  auto candidates = graph.tape();
  // Visit consumers first to avoid repeatedly expanding new input lists.
  std::reverse(candidates.begin(), candidates.end());
  for (const auto& node : candidates) {
    auto* concat = dynamic_cast<Concatenate*>(node.primitive_ptr().get());
    if (!graph.contains(node) || !concat) {
      continue;
    }
    auto axis = concat->state();
    auto stream = concat->stream();
    std::unordered_map<uintptr_t, std::vector<array>> replacements;
    for (const auto& input : node.inputs()) {
      auto* inner = dynamic_cast<Concatenate*>(input.primitive_ptr().get());
      if (!inner || inner->state() != axis ||
          replacements.contains(input.id())) {
        continue;
      }
      if (inner->stream() != stream) {
        log_skip(context, input, "stream_mismatch");
        continue;
      }
      std::vector<array> slices;
      bool exclusive = !graph.is_output(input);
      for (const auto& [use, _] : graph.consumers().at(input.id())) {
        if (use.id() == node.id()) {
          continue;
        }
        auto* selection = dynamic_cast<Slice*>(use.primitive_ptr().get());
        if (!selection || selection->stream() != stream ||
            graph.is_output(use) ||
            std::any_of(
                graph.consumers().at(use.id()).begin(),
                graph.consumers().at(use.id()).end(),
                [&](const auto& consumer) {
                  return consumer.first.id() != node.id();
                })) {
          exclusive = false;
          break;
        }
        auto [start, stop, strides] = selection->state();
        for (int d = 0; d < input.ndim(); ++d) {
          exclusive &= strides[d] == 1 &&
              (d == axis || (start[d] == 0 && stop[d] == input.shape(d)));
        }
        exclusive &= 0 <= start[axis] && start[axis] <= stop[axis] &&
            stop[axis] <= input.shape(axis);
        slices.push_back(use);
      }
      if (!exclusive) {
        log_skip(context, input, "concat_not_removable");
        continue;
      }
      bool proven = true;
      for (int d = 0; d < input.ndim(); ++d) {
        proven &=
            context.dimension_is(input, d, input.shape(d)) == ShapeCheck::True;
      }
      Shape boundaries{0};
      for (const auto& part : input.inputs()) {
        proven &= context.dimension_is(part, axis, part.shape(axis)) ==
            ShapeCheck::True;
        boundaries.push_back(boundaries.back() + part.shape(axis));
      }
      if (!proven) {
        log_skip(context, input, "segment_boundaries_not_proven");
        continue;
      }
      replacements.emplace(input.id(), input.inputs());
      for (const auto& selection : slices) {
        if (replacements.contains(selection.id())) {
          continue;
        }
        auto [start, stop, strides] =
            static_cast<Slice&>(selection.primitive()).state();
        auto& parts = replacements[selection.id()];
        auto first =
            std::upper_bound(boundaries.begin(), boundaries.end(), start[axis]);
        size_t i = first - boundaries.begin() - 1;
        for (; i < input.inputs().size() && boundaries[i] < stop[axis]; ++i) {
          const auto& part = input.inputs()[i];
          auto begin = std::max(start[axis], boundaries[i]) - boundaries[i];
          auto end = std::min(stop[axis], boundaries[i + 1]) - boundaries[i];
          if (begin == end) {
            continue;
          }
          Shape part_start(part.ndim(), 0);
          auto part_stop = part.shape();
          part_start[axis] = begin;
          part_stop[axis] = end;
          parts.push_back(slice(part, part_start, part_stop, stream));
        }
      }
    }
    if (replacements.empty()) {
      continue;
    }
    std::vector<array> inputs;
    for (const auto& input : node.inputs()) {
      if (auto it = replacements.find(input.id()); it != replacements.end()) {
        inputs.insert(inputs.end(), it->second.begin(), it->second.end());
      } else {
        inputs.push_back(input);
      }
    }
    auto output = concatenate(std::move(inputs), axis, stream);
    if (context.same_shape(node, output) == ShapeCheck::True) {
      graph.replace_outputs({node}, {output});
      context.log_rewrite("concatenate_slices", node, output);
    }
  }
}

void fold_uniform_values(CompileGraph& graph, const CompileContext& context) {
  std::unordered_map<uintptr_t, array> scalars;
  std::unordered_set<uintptr_t> input_ids;
  for (const auto& input : graph.inputs()) {
    input_ids.insert(input.id());
  }
  auto uniform = [&](const array& a) -> std::optional<array> {
    if (auto it = scalars.find(a.id()); it != scalars.end()) {
      return it->second;
    }
    return a.ndim() == 0 ? std::optional<array>(a) : std::nullopt;
  };
  auto boolean = [&](const array& a) -> std::optional<bool> {
    if (a.dtype() == bool_ && a.is_available() && !a.has_primitive() &&
        !input_ids.contains(a.id())) {
      return *a.data<bool>();
    }
    return std::nullopt;
  };

  auto candidates = graph.tape();
  for (const auto& node : candidates) {
    if (!graph.contains(node) || !node.has_primitive() ||
        node.inputs().empty() || input_ids.contains(node.id())) {
      continue;
    }
    auto first = uniform(node.inputs()[0]);
    if (!first) {
      continue;
    }
    auto& p = node.primitive();
    bool preserves = preserves_uniform_value(p);
    if (!preserves && !is_pointwise(p)) {
      continue;
    }
    if (std::any_of(
            node.inputs().begin(), node.inputs().end(), [&](const auto& in) {
              return in.has_primitive() &&
                  in.primitive().stream() != p.stream();
            })) {
      log_skip(context, node, "stream_mismatch");
      continue;
    }
    if (typeid(p) == typeid(Broadcast) && node.inputs().size() == 1 &&
        node.inputs()[0].ndim() == 0) {
      scalars.emplace(node.id(), *first);
      continue;
    }
    if (context.size_at_most(*first, node) != ShapeCheck::True) {
      log_skip(context, node, "nonempty_shape_not_proven");
      continue;
    }
    if (typeid(p) == typeid(Select)) {
      if (auto condition = boolean(*first)) {
        auto selected = node.inputs()[*condition ? 1 : 2];
        if (context.same_shape(node, selected) == ShapeCheck::True) {
          graph.replace_outputs({node}, {selected});
          context.log_rewrite("uniform_select", node, selected);
          continue;
        }
      }
    }
    std::vector<array> inputs{*first};
    if (!preserves) {
      for (size_t i = 1; i < node.inputs().size(); ++i) {
        auto scalar = uniform(node.inputs()[i]);
        if (!scalar) {
          break;
        }
        inputs.push_back(*scalar);
      }
      if (inputs.size() != node.inputs().size()) {
        continue;
      }
    }
    auto scalar = *first;
    if (!preserves) {
      std::optional<bool> folded;
      if (node.dtype() == bool_) {
        std::vector<std::optional<bool>> values;
        for (const auto& in : inputs) {
          values.push_back(boolean(in));
        }
        folded = fold_boolean(p, values);
      }
      if (folded) {
        scalar = array(*folded);
      } else if (node.ndim() == 0) {
        continue;
      } else {
        scalar =
            array({}, node.dtype(), node.primitive_ptr(), std::move(inputs));
      }
    }
    auto output = broadcast_to(scalar, node.shape(), p.stream());
    graph.replace_outputs({node}, {output});
    scalars.insert_or_assign(output.id(), scalar);
    context.log_rewrite("uniform_value", node, output);
  }
}

void eliminate_algebraic_identities(
    CompileGraph& graph,
    const CompileContext& context) {
  std::unordered_set<uintptr_t> input_ids;
  for (const auto& input : graph.inputs()) {
    input_ids.insert(input.id());
  }
  auto candidates = graph.tape();
  for (const auto& node : candidates) {
    if (!graph.contains(node) || !node.has_primitive() ||
        node.inputs().size() != 2) {
      continue;
    }
    auto& p = node.primitive();
    bool one = typeid(p) == typeid(Multiply) || typeid(p) == typeid(LogicalAnd);
    bool subtract = typeid(p) == typeid(Subtract);
    bool arithmetic = issubdtype(node.dtype(), integer) &&
        (typeid(p) == typeid(Add) || subtract || typeid(p) == typeid(Multiply));
    bool logical = node.dtype() == bool_ &&
        (typeid(p) == typeid(LogicalAnd) || typeid(p) == typeid(LogicalOr));
    if (!arithmetic && !logical) {
      continue;
    }
    int retained = -1;
    if (is_identity_constant(node.inputs()[1], one, input_ids, p.stream())) {
      retained = 0;
    } else if (
        !subtract &&
        is_identity_constant(node.inputs()[0], one, input_ids, p.stream())) {
      retained = 1;
    }
    if (retained < 0) {
      log_skip(context, node, "constant_value_not_proven");
      continue;
    }
    auto output = node.inputs()[retained];
    if (output.has_primitive() && output.primitive().stream() != p.stream()) {
      log_skip(context, node, "stream_mismatch");
    } else if (context.same_shape(node, output) != ShapeCheck::True) {
      log_skip(context, node, "shape_not_proven");
    } else {
      graph.replace_outputs({node}, {output});
      context.log_rewrite("algebraic_identity", node, output);
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

} // namespace mlx::core::detail
