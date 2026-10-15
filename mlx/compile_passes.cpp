// Copyright © 2026 Apple Inc.

#include <optional>

#include <fmt/format.h>

#include "mlx/compile_impl.h"
#include "mlx/transforms_impl.h"

namespace mlx::core::detail {

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
      {"eliminate_concat_split",
       eliminate_concat_split,
       PassScope::CompileAndExport},
      {"combine_concatenate_slices",
       combine_concatenate_slices,
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
      {"eliminate_redundant_casts",
       eliminate_redundant_casts,
       PassScope::CompileAndExport},
      {"fold_uniform_values", fold_uniform_values, PassScope::CompileAndExport},
      {"eliminate_algebraic_identities",
       eliminate_algebraic_identities,
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
