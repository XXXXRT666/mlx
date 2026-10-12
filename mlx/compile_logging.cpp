// Copyright © 2026 Apple Inc.

#include <atomic>
#include <cstdlib>

#include <fmt/format.h>

#include "mlx/compile_logging.h"

namespace mlx::core::detail {

std::string_view compile_log_category_name(CompileLogCategory category) {
  static constexpr std::string_view names[] = {
      "passes", "fusion", "guards", "recompiles", "codegen"};
  return names[static_cast<unsigned>(category)];
}

uint64_t next_compilation_id() {
  static std::atomic<uint64_t> next_id{0};
  return next_id.fetch_add(1, std::memory_order_relaxed) + 1;
}

CompileLog::CompileLog(uint64_t compilation_id)
    : compilation_id(compilation_id) {
  if (const char* value = std::getenv("MLX_COMPILE_LOGS")) {
    std::string_view tokens(value);
    while (!tokens.empty()) {
      auto end = tokens.find(',');
      auto token = tokens.substr(0, end);
      for (auto category :
           {CompileLogCategory::Passes,
            CompileLogCategory::Fusion,
            CompileLogCategory::Guards,
            CompileLogCategory::Recompiles,
            CompileLogCategory::Codegen}) {
        if (token == compile_log_category_name(category)) {
          categories |= 1 << static_cast<unsigned>(category);
        }
      }
      if (end == std::string_view::npos) {
        break;
      }
      tokens.remove_prefix(end + 1);
    }
  }
}

void CompileLog::emit(CompileLogCategory category, std::string_view message)
    const {
  fmt::print(
      stderr,
      "[compile:{}] [{}] stage={} pass={} {}\n",
      compilation_id,
      compile_log_category_name(category),
      stage,
      pass,
      message);
}

} // namespace mlx::core::detail
