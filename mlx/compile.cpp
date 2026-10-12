// Copyright © 2023-2024 Apple Inc.

#include <atomic>
#include <cstdlib>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "mlx/allocator.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/compile.h"
#include "mlx/compile_impl.h"
#include "mlx/fast_primitives.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"

namespace mlx::core {

bool is_reduction(const Primitive& p) {
  return typeid(p) == typeid(Reduce) || typeid(p) == typeid(ArgReduce);
}

Compiled::Compiled(
    Stream stream,
    std::vector<array> inputs,
    std::vector<array> outputs,
    std::vector<array> tape,
    std::unordered_set<uintptr_t> constant_ids,
    uint64_t compilation_id)
    : Primitive(stream),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      tape_(std::move(tape)),
      constant_ids_(std::move(constant_ids)),
      is_constant_([this](size_t i) {
        return constant_ids_.find(inputs_[i].id()) != constant_ids_.end();
      }),
      compilation_id_(compilation_id) {
  // Build the kernel name.
  NodeNamer namer;
  std::ostringstream os;
  std::ostringstream constant_hasher;

  std::unordered_set<uintptr_t> output_ids;
  for (auto& o : outputs_) {
    output_ids.insert(o.id());
  }

  // Fill the input names. This is not really necessary, I just like having A,
  // B, C, ... as the inputs.
  for (const auto& x : inputs_) {
    namer.get_name(x);
  }

  // The primitives describing the tape. For unary and binary primitives this
  // must be enough to describe the full computation.
  for (const auto& a : tape_) {
    // name and type of output
    os << namer.get_name(a) << kindof(a.dtype()) << a.itemsize();
    // whether or not it's an output
    if (output_ids.find(a.id()) != output_ids.end()) {
      os << "O";
    } else {
      os << "I";
    }
    // computation performed
    os << a.primitive().name();
    // name of inputs to the function
    for (auto& inp : a.inputs()) {
      os << namer.get_name(inp);
    }
  }
  os << "_";

  for (const auto& x : inputs_) {
    if (constant_ids_.find(x.id()) != constant_ids_.end()) {
      os << "C";
      print_constant(constant_hasher, x);
    } else {
      os << (is_scalar(x) ? "S" : "V");
    }
  }
  os << "_";
  // Iterate the moved-into members; the parameters are moved-from above.
  for (const auto& x : inputs_) {
    if (constant_ids_.find(x.id()) != constant_ids_.end()) {
      continue;
    }
    os << kindof(x.dtype()) << x.itemsize();
  }
  os << "_" << std::hash<std::string>{}(constant_hasher.str());

  kernel_lib_ = os.str();
}

std::vector<array> Compiled::vjp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("[Compiled] Cannot vjp primitive.");
}

std::vector<array> Compiled::jvp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[Compiled] Cannot jvp primitive.");
}

std::pair<std::vector<array>, std::vector<int>> Compiled::vmap(
    const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[Compiled] Cannot vmap primitive.");
}

bool Compiled::is_equivalent(const Primitive& other) const {
  const Compiled& a_other = static_cast<const Compiled&>(other);
  return std::equal(
      tape_.begin(),
      tape_.end(),
      a_other.tape_.begin(),
      a_other.tape_.end(),
      [](const array& a1, const array& a2) {
        auto& p1 = a1.primitive();
        auto& p2 = a2.primitive();
        return typeid(p1) == typeid(p2) && p1.is_equivalent(p2);
      });
}

const char* Compiled::name() const {
  if (name_.empty()) {
    std::ostringstream os;
    os << "Compiled";
    for (auto& a : tape_) {
      os << a.primitive().name();
    }
    name_ = os.str();
  }
  return name_.c_str();
}

std::vector<Shape> Compiled::output_shapes(const std::vector<array>& inputs) {
  size_t nd = 0;
  for (auto& in : inputs) {
    nd = std::max(nd, in.ndim());
  }
  Shape out_shape(nd, 0);
  for (auto& in : inputs) {
    auto dd = nd - in.ndim();
    for (auto i = dd; i < nd; ++i) {
      out_shape[i] = std::max(out_shape[i], in.shape()[i - dd]);
    }
  }
  // All outputs have the same shape
  return std::vector<Shape>(outputs_.size(), out_shape);
}

namespace detail {

std::atomic<CompileMode>& compile_mode() {
  auto get_val = []() {
    if (std::getenv("MLX_DISABLE_COMPILE")) {
      return CompileMode::disabled;
    } else {
      return CompileMode::enabled;
    }
  };
  static std::atomic<CompileMode> compile_mode_ = get_val();
  return compile_mode_;
}

template <typename T, typename... U>
std::uintptr_t get_function_address(const std::function<T(U...)>& fun) {
  using FunType = T (*)(U...);
  const FunType* fun_ptr = fun.template target<FunType>();
  if (fun_ptr == nullptr) {
    return 0;
  }
  return reinterpret_cast<std::uintptr_t>(*fun_ptr);
}

class CompileCache {
 public:
  struct CacheEntry {
    CacheEntry(std::optional<Stream> stream, bool shapeless)
        : stream(stream), shapeless(shapeless) {}
    std::optional<Stream> stream;
    uint64_t compilation_id{next_compilation_id()};
    bool shapeless;
    std::vector<array> inputs;
    std::vector<array> outputs;
    std::vector<array> tape;
    bool empty{true};
    std::vector<uint64_t> constants;
    std::shared_ptr<void> extra;
  };

  CompileCache() {
    // Make sure the allocator is fully initialized before the compiler cache.
    allocator::allocator();
  }

  // Returns a reference to a CacheEntry which can be updated by the caller to
  // avoid copying large tapes / inputs / outputs, with the shared_ptr of
  // entries to avoid getting erased during compilation.
  std::tuple<CacheEntry&, std::shared_ptr<std::vector<CacheEntry>>> find(
      std::uintptr_t fun_id,
      const std::vector<array>& inputs,
      bool shapeless,
      const std::vector<uint64_t>& constants,
      const CompileLog& log) {
    // Find the cache entries for |fun_id| in a thread-safe way.
    auto entries_ptr = [&]() {
      // Lookup with shared lock.
      {
        std::shared_lock lock(mutex_);
        auto it = cache_.find(fun_id);
        if (it != cache_.end()) {
          return it->second;
        }
      }
      // Insertion with exclusive lock.
      std::unique_lock lock(mutex_);
      auto& ptr = cache_[fun_id];
      if (!ptr) {
        ptr = std::make_shared<std::vector<CacheEntry>>();
      }
      return ptr;
    }();
    auto& entries = *entries_ptr;

    auto stream = peek_default_stream(default_device());
    for (CacheEntry& entry : entries) {
      auto entry_log = log;
      entry_log.compilation_id = entry.compilation_id;
      auto check = [&](bool matches,
                       std::string_view condition,
                       auto&& expected,
                       auto&& actual,
                       int input = -1) {
        if (!matches) {
          entry_log.write(CompileLogCategory::Recompiles, [&] {
            return fmt::format(
                "event=reject function={} condition={} input={}",
                fun_id,
                condition,
                input);
          });
          entry_log.write(CompileLogCategory::Guards, [&] {
            return fmt::format(
                "check_stage=call check={} input={} result=False expected={} actual={}",
                condition,
                input,
                std::forward<decltype(expected)>(expected),
                std::forward<decltype(actual)>(actual));
          });
        }
        return matches;
      };
      auto stream_key = [](const std::optional<Stream>& s) {
        return s
            ? std::tuple(
                  static_cast<int>(s->device.type), s->device.index, s->index)
            : std::tuple(-1, -1, -1);
      };
      if (!check(
              entry.stream == stream,
              "stream_device",
              stream_key(entry.stream),
              stream_key(stream)) ||
          !check(
              entry.shapeless == shapeless,
              "shapeless",
              entry.shapeless,
              shapeless) ||
          !check(
              entry.inputs.size() == inputs.size(),
              "arity",
              entry.inputs.size(),
              inputs.size())) {
        continue;
      }
      bool matches = true;
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& expected = entry.inputs[i];
        const auto& actual = inputs[i];
        if (!check(
                expected.ndim() == actual.ndim(),
                "rank",
                expected.ndim(),
                actual.ndim(),
                i) ||
            (!shapeless &&
             !check(
                 expected.shape() == actual.shape(),
                 "shape",
                 fmt::join(expected.shape(), ","),
                 fmt::join(actual.shape(), ","),
                 i)) ||
            !check(
                expected.dtype() == actual.dtype(),
                "dtype",
                fmt::streamed(expected.dtype()),
                fmt::streamed(actual.dtype()),
                i)) {
          matches = false;
          break;
        }
      }
      if (!matches ||
          !check(
              entry.constants == constants,
              "constants",
              fmt::join(entry.constants, ","),
              fmt::join(constants, ","))) {
        continue;
      }
      entry_log.write(CompileLogCategory::Recompiles, [&] {
        return fmt::format(
            "event={} function={} reason={}",
            entry.empty ? "retry" : "hit",
            fun_id,
            entry.empty ? "empty_entry" : "signature_match");
      });
      entry_log.write(CompileLogCategory::Guards, [&] {
        return fmt::format(
            "check_stage=call check=cache_signature result=True contract={}",
            shapeless ? "shapeless" : "static");
      });
      return {entry, std::move(entries_ptr)};
    }
    entries.push_back(CacheEntry{stream, shapeless});
    auto entry_log = log;
    entry_log.compilation_id = entries.back().compilation_id;
    entry_log.write(CompileLogCategory::Recompiles, [&] {
      return fmt::format(
          "event=miss function={} reason={}",
          fun_id,
          entries.size() == 1 ? "cold" : "no_matching_entry");
    });
    return {entries.back(), std::move(entries_ptr)};
  }

  void erase(std::uintptr_t fun_id) {
    std::unique_lock lock(mutex_);
    cache_.erase(fun_id);
  }

  void clear() {
    std::unique_lock lock(mutex_);
    cache_.clear();
  }

 private:
  // The cache may get its key erased from a separate thread, but its value is
  // only added and modified in the thread of creation.
  // Put value in a shared_ptr to avoid race condition when erasing happened
  // during compilation for the same function.
  std::unordered_map<std::uintptr_t, std::shared_ptr<std::vector<CacheEntry>>>
      cache_;
  std::shared_mutex mutex_;
};

std::shared_ptr<CompileCache>& compile_cache_unsafe() {
  static thread_local auto cache = std::make_shared<CompileCache>();
  return cache;
}

std::tuple<std::vector<array>, std::vector<array>, std::shared_ptr<void>>
compile_trace(
    const ArrayFnWithExtra& fun,
    const std::vector<array>& inputs,
    bool shapeless) {
  // Set the global tracing flag.
  detail::InTracing in_tracing{shapeless};

  // Run the function on placeholder inputs
  // to get compute graph
  std::vector<array> tracer_inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    array in(inputs[i].shape(), inputs[i].dtype(), nullptr, {});
    in.set_tracer(true);
    tracer_inputs.push_back(std::move(in));
  }

  auto output = fun(tracer_inputs);
  return {tracer_inputs, output.first, output.second};
}

std::vector<array> compile_replace(
    const std::vector<array>& tape,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    const std::vector<array>& inputs,
    bool shapeless) {
  std::unordered_map<uintptr_t, array> trace_to_real;
  for (int i = 0; i < inputs.size(); ++i) {
    trace_to_real.insert({trace_inputs[i].id(), inputs[i]});
  }

  auto is_load = [](const Primitive& p) { return typeid(p) == typeid(Load); };

  for (auto& a : tape) {
    // Arrays in the tape without primitives are either:
    // - inputs, which are already in the map
    // - constants, which can be used directly
    // - a load primitive which has no inputs and will become a constant
    //   after the first eval
    if (!a.has_primitive() || is_load(a.primitive())) {
      trace_to_real.insert({a.id(), a});
    } else {
      // Find real inputs
      std::vector<array> real_inputs;
      for (auto& in : a.inputs()) {
        real_inputs.push_back(trace_to_real.at(in.id()));
      }
      if (a.siblings().empty()) {
        auto shape =
            shapeless ? a.primitive().output_shapes(real_inputs)[0] : a.shape();
        auto real_a = array(
            std::move(shape),
            a.dtype(),
            a.primitive_ptr(),
            std::move(real_inputs));
        trace_to_real.insert({a.id(), std::move(real_a)});
      } else {
        // Ensure the order is correct for multi-output primitives
        std::vector<Dtype> types;
        auto trace_out = a.outputs();
        for (auto& o : trace_out) {
          types.push_back(o.dtype());
        }
        std::vector<Shape> shapes;
        if (shapeless) {
          shapes = a.primitive().output_shapes(real_inputs);
        } else {
          for (auto& o : trace_out) {
            shapes.push_back(o.shape());
          }
        }
        auto real_out = array::make_arrays(
            std::move(shapes), types, a.primitive_ptr(), real_inputs);
        for (int i = 0; i < trace_out.size(); ++i) {
          trace_to_real.insert({trace_out[i].id(), std::move(real_out[i])});
        }
      }
    }
  }

  std::vector<array> outputs;
  for (auto& o : trace_outputs) {
    outputs.push_back(trace_to_real.at(o.id()));
  }
  return outputs;
}

bool skip_compile() {
  return compile_mode() == CompileMode::disabled ||
      !(compile_available_for_device(default_device()));
}

ArrayFnWithExtra compile(
    ArrayFnWithExtra fun,
    std::uintptr_t fun_id,
    bool shapeless /* = false */,
    std::vector<uint64_t> constants /* = {} */) {
  if (skip_compile()) {
    return fun;
  }
  if (!fun) {
    throw std::invalid_argument(
        "[compile] Cannot compile a function without a target.");
  }

  return [fun = std::move(fun),
          fun_id,
          shapeless,
          constants = std::move(constants)](const std::vector<array>& inputs) {
    // If the inputs are tracers, trace the original graph
    if (std::any_of(inputs.begin(), inputs.end(), [](auto& in) {
          return in.is_tracer();
        })) {
      return fun(inputs);
    }

    // Find a cache entry with the correct inputs
    CompileLog log;
    log.stage = "cache";
    auto [entry, entries_ptr] =
        compile_cache_unsafe()->find(fun_id, inputs, shapeless, constants, log);
    log.compilation_id = entry.compilation_id;
    static_assert(std::is_reference_v<decltype(entry)>);

    // No matching cache entry existed, so compile
    if (entry.empty) {
      // Set the constants
      entry.constants = std::move(constants);
      // Trace to build the graph
      std::tie(entry.inputs, entry.outputs, entry.extra) =
          compile_trace(fun, inputs, shapeless);

      CompileGraph graph(entry.inputs, entry.outputs, inputs);
      compile_optimize(
          graph,
          CompileContext(
              shapeless, CompileTarget::Compile, compile_mode(), log));
      entry.outputs = graph.outputs();
      entry.tape = graph.tape();

      // Mark the entry as filled only after every step above completed, so
      // a throwing first trace leaves the entry empty and a later call
      // re-traces cleanly instead of hitting a half-filled entry
      entry.empty = false;
    }

    // At this point we must have a tape, now replace the placeholders
    // with real arrays that can be evaluated
    return ArraysAndExtra{
        compile_replace(
            entry.tape, entry.inputs, entry.outputs, inputs, shapeless),
        entry.extra};
  };
}

std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    std::uintptr_t fun_id,
    bool shapeless /* = false */,
    std::vector<uint64_t> constants /* = {} */) {
  if (skip_compile()) {
    return fun;
  }
  if (!fun) {
    throw std::invalid_argument(
        "[compile] Cannot compile a function without a target.");
  }

  ArrayFnWithExtra fun_with_extra =
      [fun = std::move(fun)](const std::vector<array>& inputs) {
        return ArraysAndExtra{fun(inputs), nullptr};
      };

  auto compiled_fun = compile(
      std::move(fun_with_extra), fun_id, shapeless, std::move(constants));

  return [compiled_fun =
              std::move(compiled_fun)](const std::vector<array>& inputs) {
    return compiled_fun(inputs).first;
  };
}

CompileCacheWeakPtr compile_cache() {
  return compile_cache_unsafe();
}

void compile_erase(const CompileCacheWeakPtr& cache, std::uintptr_t fun_id) {
  if (auto p = cache.lock()) {
    p->erase(fun_id);
  }
}

void compile_clear_cache(const CompileCacheWeakPtr& cache) {
  if (auto p = cache.lock()) {
    p->clear();
  }
}

} // namespace detail

std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    bool shapeless /* false */) {
  if (detail::skip_compile()) {
    return fun;
  }
  auto fun_id = detail::get_function_address(fun);
  if (fun_id) {
    // If the function has an addressable target then no need to manage it's
    // lifetime
    return detail::compile(std::move(fun), fun_id, shapeless);
  } else {
    auto pfun = std::shared_ptr<
        std::function<std::vector<array>(const std::vector<array>&)>>(
        new std::function<std::vector<array>(const std::vector<array>&)>{fun},
        [cache = detail::compile_cache()](auto* p) {
          detail::compile_erase(cache, reinterpret_cast<std::uintptr_t>(p));
          delete p;
        });
    fun_id = reinterpret_cast<std::uintptr_t>(pfun.get());
    return detail::compile(
        [pfun = std::move(pfun)](const auto& inputs) {
          return (*pfun)(inputs);
        },
        fun_id,
        shapeless);
  }
}

std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::vector<array> (*fun)(const std::vector<array>&),
    bool shapeless /* = false */) {
  if (detail::skip_compile()) {
    return fun;
  }
  return detail::compile(fun, reinterpret_cast<std::uintptr_t>(fun), shapeless);
}

void disable_compile() {
  detail::compile_mode() = CompileMode::disabled;
}

void enable_compile() {
  detail::compile_mode() = CompileMode::enabled;
}

void set_compile_mode(CompileMode mode) {
  detail::compile_mode() = mode;
}

} // namespace mlx::core
