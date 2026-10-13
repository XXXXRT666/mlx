// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <optional>

#include <fmt/format.h>

#include "mlx/compile_impl.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core::detail {
namespace {

bool is_view(const Primitive& p) {
  return typeid(p) == typeid(Reshape) || typeid(p) == typeid(Transpose) ||
      typeid(p) == typeid(Flatten) || typeid(p) == typeid(Unflatten) ||
      typeid(p) == typeid(Squeeze) || typeid(p) == typeid(ExpandDims) ||
      typeid(p) == typeid(Broadcast);
}

bool is_selection(const Primitive& p) {
  return typeid(p) == typeid(Slice) || typeid(p) == typeid(Gather) ||
      typeid(p) == typeid(GatherAxis);
}

bool used_only_by(
    const CompileGraph& graph,
    const array& a,
    const array& consumer) {
  const auto& uses = graph.consumers().at(a.id());
  return !graph.is_output(a) &&
      std::all_of(uses.begin(), uses.end(), [&](const auto& use) {
        return use.first.id() == consumer.id();
      });
}

std::optional<array> uniform_scalar(const array& a) {
  if (a.ndim() == 0) {
    return a;
  }
  if (a.has_primitive()) {
    auto& p = a.primitive();
    if (typeid(p) == typeid(Broadcast) && a.inputs().size() == 1 &&
        a.inputs()[0].ndim() == 0) {
      return a.inputs()[0];
    }
  }
  return std::nullopt;
}

void log_skip(
    const CompileContext& context,
    const array& a,
    std::string_view reason) {
  context.log.write(CompileLogCategory::Guards, [&] {
    return fmt::format(
        "rule={} check_stage=compile node={} reason={}",
        context.log.pass,
        a.id(),
        reason);
  });
}

} // namespace

void reorder_pointwise_views(
    CompileGraph& graph,
    const CompileContext& context) {
  struct ViewChain {
    array node;
    std::shared_ptr<const ViewChain> input;
  };
  struct Pending {
    array value;
    std::shared_ptr<const ViewChain> views;
    bool changed;
  };
  std::unordered_map<uintptr_t, Pending> pending;
  auto materialize = [&](const array& node) {
    auto it = pending.find(node.id());
    if (it == pending.end()) {
      return;
    }
    if (it->second.changed) {
      auto output = it->second.value;
      std::vector<array> views;
      for (auto link = it->second.views; link; link = link->input) {
        views.push_back(link->node);
      }
      for (auto view = views.rbegin(); view != views.rend(); ++view) {
        output = array(
            view->shape(), output.dtype(), view->primitive_ptr(), {output});
      }
      graph.replace_outputs({node}, {output});
      context.log_rewrite("pointwise_before_view", node, output);
    }
    pending.erase(it);
  };
  auto same_views = [](auto a, auto b) {
    while (a != b) {
      if (!a || !b) {
        return false;
      }
      auto& pa = a->node.primitive();
      auto& pb = b->node.primitive();
      if (typeid(pa) != typeid(pb) || !pa.is_equivalent(pb)) {
        return false;
      }
      a = a->input;
      b = b->input;
    }
    return true;
  };

  auto candidates = graph.tape();
  for (const auto& node : candidates) {
    if (!graph.contains(node) || !node.has_primitive()) {
      continue;
    }
    bool view_node = node.inputs().size() == 1 && is_view(node.primitive());
    if (pending.empty() && !view_node) {
      continue;
    }
    auto stream = node.primitive().stream();
    for (const auto& in : node.inputs()) {
      auto it = pending.find(in.id());
      if (it != pending.end() &&
          (it->second.views->node.primitive().stream() != stream ||
           !used_only_by(graph, in, node))) {
        log_skip(context, node, "view_stream_mismatch_or_shared");
        auto original = in;
        materialize(original);
      }
    }
    if (view_node) {
      auto input = node.inputs()[0];
      auto it = pending.find(input.id());
      Pending value =
          it == pending.end() ? Pending{input, nullptr, false} : it->second;
      value.views = std::make_shared<ViewChain>(ViewChain{node, value.views});
      pending.emplace(node.id(), std::move(value));
      continue;
    }

    const Pending* view = nullptr;
    for (const auto& in : node.inputs()) {
      if (auto it = pending.find(in.id()); it != pending.end()) {
        view = &it->second;
        if (view->value.ndim() != 0) {
          break;
        }
      }
    }
    if (!view) {
      continue;
    }
    std::vector<array> inputs;
    if (is_pointwise(node.primitive())) {
      for (const auto& in : node.inputs()) {
        auto it = pending.find(in.id());
        if (it != pending.end() && same_views(it->second.views, view->views) &&
            context.same_shape(it->second.value, view->value) ==
                ShapeCheck::True) {
          inputs.push_back(it->second.value);
          continue;
        }
        auto scalar = it != pending.end() && it->second.value.ndim() == 0
            ? std::optional<array>(it->second.value)
            : uniform_scalar(in);
        if (scalar && !context.shapeless &&
            (!in.has_primitive() || in.primitive().stream() == stream)) {
          inputs.push_back(broadcast_to(*scalar, view->value.shape(), stream));
          continue;
        }
        break;
      }
    }
    if (inputs.size() == node.inputs().size()) {
      auto value = array(
          view->value.shape(),
          node.dtype(),
          node.primitive_ptr(),
          std::move(inputs));
      pending.emplace(node.id(), Pending{value, view->views, true});
    } else {
      log_skip(context, node, "view_mapping_not_proven");
      inputs = node.inputs();
      for (const auto& in : inputs) {
        materialize(in);
      }
    }
  }
  // Keep view chains deferred until a boundary, so each chain is built once.
  auto outputs = graph.outputs();
  for (const auto& out : outputs) {
    materialize(out);
  }
}

void reorder_pointwise_selections(
    CompileGraph& graph,
    const CompileContext& context) {
  auto candidates = graph.tape();
  while (!candidates.empty()) {
    auto node = candidates.back();
    candidates.pop_back();
    if (!graph.contains(node) || !node.has_primitive() ||
        !is_selection(node.primitive())) {
      continue;
    }
    auto pointwise = node.inputs()[0];
    if (!pointwise.has_primitive() || !is_pointwise(pointwise.primitive())) {
      continue;
    }
    if (node.primitive().stream() != pointwise.primitive().stream() ||
        !used_only_by(graph, pointwise, node) ||
        context.size_at_most(node, pointwise) != ShapeCheck::True ||
        !std::all_of(
            pointwise.inputs().begin(),
            pointwise.inputs().end(),
            [&](const auto& in) {
              return context.same_shape(in, pointwise) == ShapeCheck::True;
            })) {
      log_skip(context, node, "selection_domain_not_proven_or_shared");
      continue;
    }
    std::vector<array> inputs;
    for (const auto& in : pointwise.inputs()) {
      auto selected_inputs = node.inputs();
      selected_inputs[0] = in;
      inputs.emplace_back(
          node.shape(),
          in.dtype(),
          node.primitive_ptr(),
          std::move(selected_inputs));
    }
    auto output =
        array(node.shape(), node.dtype(), pointwise.primitive_ptr(), inputs);
    graph.replace_outputs({node}, {output});
    context.log_rewrite("selection_before_pointwise", node, output);
    // Each selection moves toward the leaves through a pointwise producer.
    candidates.insert(candidates.end(), inputs.begin(), inputs.end());
  }
}

void reorder_pointwise_concatenations(
    CompileGraph& graph,
    const CompileContext& context) {
  auto candidates = graph.tape();
  for (auto node : candidates) {
    // Each edit removes one common pointwise layer from the concat inputs.
    while (graph.contains(node) && node.has_primitive()) {
      auto& concat = node.primitive();
      if (typeid(concat) != typeid(Concatenate)) {
        break;
      }
      const auto& parts = node.inputs();
      const auto& first = parts[0];
      if (!first.has_primitive() || !is_pointwise(first.primitive())) {
        break;
      }
      auto& p = first.primitive();
      if (!std::all_of(parts.begin(), parts.end(), [&](const auto& part) {
            if (!part.has_primitive()) {
              return false;
            }
            auto& q = part.primitive();
            return typeid(p) == typeid(q) && p.is_equivalent(q) &&
                q.stream() == concat.stream() &&
                used_only_by(graph, part, node) &&
                graph.consumers().at(part.id()).size() == 1 &&
                part.inputs().size() == first.inputs().size();
          })) {
        log_skip(context, node, "pointwise_calls_differ_or_shared");
        break;
      }
      std::vector<array> inputs;
      std::optional<array> joined;
      for (size_t i = 0; i < first.inputs().size(); ++i) {
        auto scalar = uniform_scalar(first.inputs()[i]);
        if (!context.shapeless && scalar &&
            std::all_of(parts.begin(), parts.end(), [&](const auto& part) {
              auto other = uniform_scalar(part.inputs()[i]);
              return other && other->id() == scalar->id() &&
                  (!part.inputs()[i].has_primitive() ||
                   part.inputs()[i].primitive().stream() == concat.stream());
            })) {
          inputs.push_back(
              broadcast_to(*scalar, node.shape(), concat.stream()));
          continue;
        }
        if (joined || first.inputs()[i].itemsize() > node.itemsize() ||
            (context.shapeless && first.inputs().size() != 1)) {
          break;
        }
        std::vector<array> column;
        for (const auto& part : parts) {
          if (part.inputs()[i].dtype() != first.inputs()[i].dtype()) {
            break;
          }
          column.push_back(part.inputs()[i]);
        }
        if (column.size() != parts.size()) {
          break;
        }
        auto dtype = column[0].dtype();
        joined =
            array(node.shape(), dtype, node.primitive_ptr(), std::move(column));
        inputs.push_back(*joined);
      }
      if (!joined || inputs.size() != first.inputs().size()) {
        log_skip(context, node, "concat_would_add_data_movement");
        break;
      }
      auto output = array(
          node.shape(), node.dtype(), first.primitive_ptr(), std::move(inputs));
      graph.replace_outputs({node}, {output});
      context.log_rewrite("pointwise_after_concat", node, output);
      node = *joined;
    }
  }
}

} // namespace mlx::core::detail
