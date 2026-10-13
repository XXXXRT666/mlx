// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <fmt/ranges.h>

#include "mlx/compile_impl.h"
#include "mlx/primitives.h"

namespace mlx::core::detail {

CompileGraph::CompileGraph(
    std::vector<array> inputs,
    std::vector<array> outputs,
    const std::vector<array>& original_inputs)
    : inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
  std::unordered_set<uintptr_t> original_ids;
  for (const auto& a : original_inputs) {
    original_ids.insert(a.id());
  }
  std::unordered_map<uintptr_t, array> replacements;
  for (const auto& a : inputs_) {
    replacements.emplace(a.id(), a);
  }
  std::unordered_set<uintptr_t> active;
  std::function<array(const array&)> isolate = [&](const array& a) -> array {
    if (original_ids.contains(a.id())) {
      throw std::invalid_argument(
          "[compile] Attempting to compile a function with uncaptured inputs is not allowed.");
    }
    if (auto it = replacements.find(a.id()); it != replacements.end()) {
      add_node(it->second);
      return it->second;
    }
    if (!a.has_primitive()) {
      replacements.emplace(a.id(), a);
      add_node(a);
      return a;
    }
    if (!active.insert(a.id()).second) {
      throw std::logic_error("[compile] Cycle in compile graph.");
    }
    std::vector<array> inputs;
    inputs.reserve(a.inputs().size());
    for (const auto& in : a.inputs()) {
      inputs.push_back(isolate(in));
    }
    if (a.siblings().empty()) {
      array copy(a.shape(), a.dtype(), a.primitive_ptr(), std::move(inputs));
      replacements.emplace(a.id(), copy);
      add_node(copy);
      active.erase(a.id());
      return copy;
    }
    auto outputs = a.outputs();
    std::vector<Shape> shapes;
    std::vector<Dtype> dtypes;
    for (const auto& out : outputs) {
      shapes.push_back(out.shape());
      dtypes.push_back(out.dtype());
    }
    auto copies = array::make_arrays(
        std::move(shapes), dtypes, a.primitive_ptr(), std::move(inputs));
    for (size_t i = 0; i < outputs.size(); ++i) {
      replacements.emplace(outputs[i].id(), copies[i]);
    }
    add_node(copies[a.sibling_position()]);
    active.erase(a.id());
    return replacements.at(a.id());
  };
  for (size_t i = 0; i < outputs_.size(); ++i) {
    outputs_[i] = isolate(outputs_[i]);
    output_positions_[outputs_[i].id()].push_back(i);
  }
  dirty_ = false;
}

const std::vector<array>& CompileGraph::tape() {
  rebuild();
  return tape_;
}

const ConsumersMap& CompileGraph::consumers() const {
  return consumers_;
}

bool CompileGraph::contains(const array& a) const {
  return depths_.contains(a.id());
}

bool CompileGraph::is_output(const array& a) const {
  return output_positions_.contains(a.id());
}

void CompileGraph::rebuild() {
  if (!dirty_) {
    return;
  }
  tape_.clear();
  std::unordered_set<uintptr_t> seen;
  std::function<void(const array&)> visit = [&](const array& a) {
    if (!seen.insert(a.id()).second) {
      return;
    }
    for (const auto& sibling : a.siblings()) {
      seen.insert(sibling.id());
    }
    for (const auto& in : a.inputs()) {
      visit(in);
    }
    tape_.push_back(a);
  };
  for (const auto& out : outputs_) {
    visit(out);
  }
  dirty_ = false;
}

void CompileGraph::add_node(const array& a) {
  if (depths_.contains(a.id())) {
    return;
  }
  size_t depth = 0;
  for (size_t i = 0; i < a.inputs().size(); ++i) {
    const auto& in = a.inputs()[i];
    add_node(in);
    depth = std::max(depth, depths_.at(in.id()) + 1);
    consumers_[in.id()].emplace_back(a, i);
    for (const auto& sibling : a.siblings()) {
      consumers_[in.id()].emplace_back(sibling, i);
    }
  }
  depths_[a.id()] = depth;
  for (const auto& sibling : a.siblings()) {
    depths_[sibling.id()] = depth;
  }
  tape_.push_back(a);
}

void CompileGraph::update_depths(std::vector<array> candidates) {
  while (!candidates.empty()) {
    auto a = candidates.back();
    candidates.pop_back();
    size_t depth = 0;
    for (const auto& in : a.inputs()) {
      depth = std::max(depth, depths_.at(in.id()) + 1);
    }
    // Upper bounds need only increase to keep every edge ordered.
    if (depth <= depths_.at(a.id())) {
      continue;
    }
    for (const auto& out : a.outputs()) {
      depths_[out.id()] = depth;
      if (auto it = consumers_.find(out.id()); it != consumers_.end()) {
        for (const auto& [consumer, _] : it->second) {
          candidates.push_back(consumer);
        }
      }
    }
  }
}

void CompileGraph::replace_outputs(
    const std::vector<array>& old_outputs,
    const std::vector<array>& new_outputs) {
  if (old_outputs.size() != new_outputs.size()) {
    throw std::invalid_argument("[compile] Replacement output count mismatch.");
  }
  if (old_outputs.empty()) {
    return;
  }
  struct Replacement {
    array next;
    std::vector<std::pair<array, int>> uses;
  };
  std::unordered_map<uintptr_t, Replacement> replacements;
  size_t min_depth = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < old_outputs.size(); ++i) {
    const auto& old = old_outputs[i];
    const auto& next = new_outputs[i];
    if (old.dtype() != next.dtype() || old.shape() != next.shape()) {
      throw std::invalid_argument(
          "[compile] Replacement output type mismatch.");
    }
    auto [it, inserted] = replacements.emplace(old.id(), Replacement{next, {}});
    if (!inserted && it->second.next.id() != next.id()) {
      throw std::invalid_argument("[compile] Conflicting output replacements.");
    }
    min_depth = std::min(min_depth, depths_.at(old.id()));
    if (auto uses = consumers_.find(old.id()); uses != consumers_.end()) {
      it->second.uses = uses->second;
    }
  }
  for (const auto& out : new_outputs) {
    add_node(out);
  }

  // Earlier dependencies cannot contain a use of a replaced value.
  std::unordered_set<uintptr_t> retained;
  std::function<void(const array&)> retain = [&](const array& a) {
    if (depths_.at(a.id()) < min_depth || !retained.insert(a.id()).second) {
      return;
    }
    for (const auto& sibling : a.siblings()) {
      retained.insert(sibling.id());
    }
    for (const auto& in : a.inputs()) {
      retain(in);
    }
  };
  for (const auto& out : new_outputs) {
    retain(out);
  }
  auto replacement = [&](const array& a) -> const array& {
    auto it = replacements.find(a.id());
    return it == replacements.end() ? a : it->second.next;
  };
  auto remove_use = [&](uintptr_t id, uintptr_t consumer_id, int index) {
    auto it = consumers_.find(id);
    if (it == consumers_.end()) {
      return;
    }
    auto& uses = it->second;
    auto use = std::find_if(uses.begin(), uses.end(), [&](const auto& entry) {
      return entry.first.id() == consumer_id && entry.second == index;
    });
    if (use != uses.end()) {
      *use = std::move(uses.back());
      uses.pop_back();
    }
    if (uses.empty()) {
      consumers_.erase(it);
    }
  };

  std::vector<std::pair<size_t, array>> roots;
  for (const auto& [id, entry] : replacements) {
    const auto& next = entry.next;
    if (id != next.id()) {
      if (auto it = output_positions_.find(id); it != output_positions_.end()) {
        for (auto index : it->second) {
          roots.emplace_back(index, next);
        }
        output_positions_.erase(it);
      }
    }
  }
  for (const auto& [index, next] : roots) {
    outputs_[index] = next;
    output_positions_[next.id()].push_back(index);
  }
  std::vector<std::tuple<array, int, array>> changes;
  for (const auto& [id, entry] : replacements) {
    const auto& next = entry.next;
    if (id == next.id()) {
      continue;
    }
    for (const auto& [consumer, index] : entry.uses) {
      if (!retained.contains(consumer.id())) {
        remove_use(id, consumer.id(), index);
        consumers_[next.id()].emplace_back(consumer, index);
        changes.emplace_back(consumer, index, next);
      }
    }
  }

  // Remove dead uses before changing arrays held by a fused tape.
  auto dead = old_outputs;
  dead.insert(dead.end(), new_outputs.begin(), new_outputs.end());
  while (!dead.empty()) {
    auto a = dead.back();
    dead.pop_back();
    if (!depths_.contains(a.id())) {
      continue;
    }
    auto is_live = [&](const auto& out) {
      return is_output(out) || consumers_.contains(out.id());
    };
    if (is_live(a) ||
        std::any_of(a.siblings().begin(), a.siblings().end(), is_live)) {
      continue;
    }
    for (size_t i = 0; i < a.inputs().size(); ++i) {
      auto in = retained.contains(a.id()) ? a.inputs()[i]
                                          : replacement(a.inputs()[i]);
      remove_use(in.id(), a.id(), i);
      for (const auto& sibling : a.siblings()) {
        remove_use(in.id(), sibling.id(), i);
      }
      dead.push_back(in);
    }
    depths_.erase(a.id());
    for (const auto& sibling : a.siblings()) {
      depths_.erase(sibling.id());
    }
  }
  std::vector<array> changed;
  for (auto& [consumer, index, next] : changes) {
    if (depths_.contains(consumer.id())) {
      consumer.inputs()[index] = next;
      changed.push_back(consumer);
    }
  }
  if (!roots.empty() || !changed.empty()) {
    ++rewrite_count_;
  }
  update_depths(std::move(changed));
  dirty_ = true;
}

array CompileGraph::split_uses(
    const array& a,
    const std::unordered_set<uintptr_t>& consumer_ids) {
  if (!a.siblings().empty()) {
    throw std::invalid_argument("[compile] Cannot split multi-output uses.");
  }
  auto id = a.id();
  array copy(a.shape(), a.dtype(), a.primitive_ptr(), a.inputs());
  add_node(copy);
  auto& selected = consumers_[copy.id()];
  auto& remaining = consumers_.at(id);
  for (size_t i = 0; i < remaining.size();) {
    auto& [consumer, index] = remaining[i];
    if (consumer_ids.contains(consumer.id())) {
      consumer.inputs()[index] = copy;
      selected.push_back(std::move(remaining[i]));
      remaining[i] = std::move(remaining.back());
      remaining.pop_back();
    } else {
      ++i;
    }
  }
  if (remaining.empty()) {
    consumers_.erase(id);
  }
  ++rewrite_count_;
  dirty_ = true;
  return copy;
}

void CompileGraph::validate() {
  dirty_ = true;
  rebuild();
  std::unordered_set<uintptr_t> seen;
  std::unordered_map<uintptr_t, std::vector<bool>> used_inputs;
  size_t edges = 0;
  for (const auto& a : tape_) {
    auto outputs = a.outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto& out = outputs[i];
      used_inputs.emplace(out.id(), std::vector<bool>(out.inputs().size()));
      if (!depths_.contains(out.id()) || out.sibling_position() != i ||
          out.primitive_id() != a.primitive_id() ||
          out.inputs().size() != a.inputs().size()) {
        throw std::logic_error("[compile] Inconsistent primitive outputs.");
      }
      auto siblings = out.outputs();
      if (siblings.size() != outputs.size()) {
        throw std::logic_error("[compile] Inconsistent sibling count.");
      }
      for (size_t j = 0; j < siblings.size(); ++j) {
        if (siblings[j].id() != outputs[j].id()) {
          throw std::logic_error("[compile] Inconsistent sibling order.");
        }
      }
      for (size_t j = 0; j < out.inputs().size(); ++j) {
        const auto& in = out.inputs()[j];
        if (!seen.contains(in.id()) || in.id() != a.inputs()[j].id()) {
          throw std::logic_error("[compile] Invalid tape dependency.");
        }
        if (depths_.at(in.id()) >= depths_.at(out.id())) {
          throw std::logic_error("[compile] Invalid dependency order.");
        }
        ++edges;
      }
    }
    for (const auto& out : outputs) {
      if (!seen.insert(out.id()).second) {
        throw std::logic_error("[compile] Duplicate primitive in tape.");
      }
    }
  }
  for (const auto& [id, uses] : consumers_) {
    for (const auto& [consumer, index] : uses) {
      if (!seen.contains(consumer.id()) || index < 0 ||
          index >= consumer.inputs().size() ||
          consumer.inputs()[index].id() != id) {
        throw std::logic_error("[compile] Invalid consumer map.");
      }
      auto& used = used_inputs.at(consumer.id());
      if (used[index]) {
        throw std::logic_error("[compile] Duplicate consumer.");
      }
      used[index] = true;
      --edges;
    }
  }
  if (edges != 0 || seen.size() != depths_.size()) {
    throw std::logic_error("[compile] Incomplete graph analyses.");
  }
  std::vector<bool> positions(outputs_.size());
  size_t num_outputs = 0;
  for (const auto& [id, indices] : output_positions_) {
    for (auto index : indices) {
      if (index >= outputs_.size() || positions[index] ||
          outputs_[index].id() != id) {
        throw std::logic_error("[compile] Invalid output positions.");
      }
      positions[index] = true;
      ++num_outputs;
    }
  }
  if (num_outputs != outputs_.size()) {
    throw std::logic_error("[compile] Missing output positions.");
  }
  for (const auto& out : outputs_) {
    if (!seen.contains(out.id())) {
      throw std::logic_error("[compile] Missing graph output.");
    }
  }
}

CompileContext::CompileContext(
    bool shapeless,
    CompileTarget target,
    CompileMode mode,
    CompileLog log)
    : shapeless(shapeless), target(target), mode(mode), log(log) {
  if (this->log.compilation_id == 0) {
    this->log.compilation_id = next_compilation_id();
  }
}

ShapeCheck CompileContext::same_shape(const array& a, const array& b) const {
  auto result = [&]() {
    if (a.ndim() != b.ndim()) {
      return ShapeCheck::False;
    }
    if (a.id() == b.id() || a.ndim() == 0) {
      return ShapeCheck::True;
    }
    if (shapeless) {
      // Equivalent calls infer the same output shape from the same inputs.
      if (a.has_primitive() && b.has_primitive() &&
          a.sibling_position() == b.sibling_position() &&
          a.inputs().size() == b.inputs().size() &&
          std::equal(
              a.inputs().begin(),
              a.inputs().end(),
              b.inputs().begin(),
              [](const auto& x, const auto& y) { return x.id() == y.id(); })) {
        const auto& pa = a.primitive();
        const auto& pb = b.primitive();
        if (typeid(pa) == typeid(pb) && pa.is_equivalent(pb)) {
          return ShapeCheck::True;
        }
      }
      return ShapeCheck::Unknown;
    }
    return a.shape() == b.shape() ? ShapeCheck::True : ShapeCheck::False;
  }();
  log.write(CompileLogCategory::Guards, [&] {
    return fmt::format(
        "check_stage=compile check=same_shape node={} other={} contract={} result={} shapes=[{}],[{}]",
        a.id(),
        b.id(),
        shapeless ? "shapeless" : "static",
        result == ShapeCheck::True        ? "True"
            : result == ShapeCheck::False ? "False"
                                          : "Unknown",
        fmt::join(a.shape(), ","),
        fmt::join(b.shape(), ","));
  });
  return result;
}

ShapeCheck CompileContext::size_at_most(const array& a, const array& b) const {
  auto result = shapeless    ? ShapeCheck::Unknown
      : a.size() <= b.size() ? ShapeCheck::True
                             : ShapeCheck::False;
  log.write(CompileLogCategory::Guards, [&] {
    return fmt::format(
        "check_stage=compile check=size_at_most node={} other={} contract={} result={}",
        a.id(),
        b.id(),
        shapeless ? "shapeless" : "static",
        result == ShapeCheck::True        ? "True"
            : result == ShapeCheck::False ? "False"
                                          : "Unknown");
  });
  return result;
}

ShapeCheck CompileContext::dimension_is(const array& a, int axis, int size)
    const {
  auto result = shapeless     ? ShapeCheck::Unknown
      : a.shape(axis) == size ? ShapeCheck::True
                              : ShapeCheck::False;
  log.write(CompileLogCategory::Guards, [&] {
    return fmt::format(
        "check_stage=compile check=dimension_is node={} axis={} expected={} shape=[{}] contract={} result={}",
        a.id(),
        axis,
        size,
        fmt::join(a.shape(), ","),
        shapeless ? "shapeless" : "static",
        result == ShapeCheck::True        ? "True"
            : result == ShapeCheck::False ? "False"
                                          : "Unknown");
  });
  return result;
}

void CompileContext::log_rewrite(
    std::string_view rule,
    const array& old_output,
    const array& new_output) const {
  log.write(CompileLogCategory::Passes, [&] {
    return fmt::format(
        "event=rewrite rule={} node={} replacement={} primitive={}",
        rule,
        old_output.id(),
        new_output.id(),
        new_output.primitive_id());
  });
}

} // namespace mlx::core::detail
