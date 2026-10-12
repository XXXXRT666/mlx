// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>
#include <string_view>

namespace mlx::core::detail {

enum class CompileLogCategory { Passes, Fusion, Guards, Recompiles, Codegen };

std::string_view compile_log_category_name(CompileLogCategory category);
uint64_t next_compilation_id();

struct CompileLog {
  explicit CompileLog(uint64_t compilation_id = 0);

  bool enabled(CompileLogCategory category) const {
    return categories & (1 << static_cast<unsigned>(category));
  }

  template <typename Message>
  void write(CompileLogCategory category, Message&& message) const {
    if (enabled(category)) {
      emit(category, message());
    }
  }

  uint64_t compilation_id;
  std::string_view stage{"-"};
  std::string_view pass{"-"};
  uint8_t categories{0};

 private:
  void emit(CompileLogCategory category, std::string_view message) const;
};

} // namespace mlx::core::detail
