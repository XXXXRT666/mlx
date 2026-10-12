// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/compile.h"

namespace mlx::core::detail {

using ArraysAndExtra = std::pair<std::vector<array>, std::shared_ptr<void>>;
using ArrayFnWithExtra =
    std::function<ArraysAndExtra(const std::vector<array>&)>;

// This is not part of the general C++ API as calling with a bad id is a bad
// idea.
MLX_API std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    std::uintptr_t fun_id,
    bool shapeless = false,
    std::vector<uint64_t> constants = {});

MLX_API ArrayFnWithExtra compile(
    ArrayFnWithExtra fun,
    std::uintptr_t fun_id,
    bool shapeless,
    std::vector<uint64_t> constants);

// Get the compiler cache of current thread.
class CompileCache;
using CompileCacheWeakPtr = std::weak_ptr<CompileCache>;
MLX_API CompileCacheWeakPtr compile_cache();

// Erase cached compile function.
MLX_API void compile_erase(
    const CompileCacheWeakPtr& cache,
    std::uintptr_t fun_id);

// Clear the compiler cache causing a recompilation of all compiled functions
// when called again.
MLX_API void compile_clear_cache(const CompileCacheWeakPtr& cache);

bool compile_available_for_device(const Device& device);

std::tuple<std::vector<array>, std::vector<array>, std::shared_ptr<void>>
compile_trace(
    const ArrayFnWithExtra& fun,
    const std::vector<array>& inputs,
    bool shapeless);

using ConsumersMap =
    std::unordered_map<std::uintptr_t, std::vector<std::pair<array, int>>>;

class CompileGraph {
 public:
  CompileGraph(
      std::vector<array> inputs,
      std::vector<array> outputs,
      const std::vector<array>& original_inputs);

  const std::vector<array>& inputs() const {
    return inputs_;
  }
  const std::vector<array>& outputs() const {
    return outputs_;
  }
  // Do not keep analysis references across graph edits.
  const std::vector<array>& tape();
  const ConsumersMap& consumers() const;
  bool contains(const array& a) const;
  bool is_output(const array& a) const;
  size_t rewrite_count() const {
    return rewrite_count_;
  }

  // Replacements are simultaneous. Rules must first prove their legality.
  void replace_outputs(
      const std::vector<array>& old_outputs,
      const std::vector<array>& new_outputs);
  array split_uses(
      const array& a,
      const std::unordered_set<uintptr_t>& consumer_ids);
  // Refresh the tape. Other graph analyses are maintained by each edit.
  void rebuild();
  void validate();

 private:
  void add_node(const array& a);
  void update_depths(std::vector<array> candidates);

  std::vector<array> inputs_;
  std::vector<array> outputs_;
  std::vector<array> tape_;
  ConsumersMap consumers_;
  // Bounds increase along each edge to prune dependency searches.
  std::unordered_map<uintptr_t, size_t> depths_;
  std::unordered_map<uintptr_t, std::vector<size_t>> output_positions_;
  size_t rewrite_count_{0};
  bool dirty_{true};
};

enum class CompileTarget { Compile, Export };
enum class ShapeCheck { True, False, Unknown };

struct CompileContext {
  CompileContext(bool shapeless, CompileTarget target, CompileMode mode);

  ShapeCheck same_shape(const array& a, const array& b) const;
  ShapeCheck dimension_is(const array& a, int axis, int size) const;
  void log_fusion(const array& a, std::string_view reason) const;

  bool shapeless;
  CompileTarget target;
  CompileMode mode;
  bool log_passes{false};
  bool log_fusions{false};
  uint64_t compilation_id{0};
  std::string_view pass_name;
};

using PassFn = void (*)(CompileGraph&, const CompileContext&);
enum class PassScope { CompileOnly, CompileAndExport };

struct PassSpec {
  std::string_view name;
  PassFn run;
  PassScope scope;
};

void compile_optimize(CompileGraph& graph, const CompileContext& context);

std::vector<array> compile_replace(
    const std::vector<array>& tape,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    const std::vector<array>& inputs,
    bool shapeless);

void compile_validate_shapeless(const std::vector<array>& tape);

} // namespace mlx::core::detail
