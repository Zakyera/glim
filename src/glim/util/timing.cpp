#include <glim/util/timing.hpp>

#include <atomic>

namespace glim {
namespace {
std::atomic_bool g_timing_enabled{false};
}

void set_timing_enabled(bool enabled) {
  g_timing_enabled.store(enabled, std::memory_order_relaxed);
}

bool timing_enabled() {
  return g_timing_enabled.load(std::memory_order_relaxed);
}

double timing_elapsed_ms(const TimingClock::time_point& start) {
  return std::chrono::duration<double, std::milli>(
             TimingClock::now() - start)
      .count();
}

}  // namespace glim
