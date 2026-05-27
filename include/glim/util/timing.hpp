#pragma once

#include <chrono>

namespace glim {

using TimingClock = std::chrono::steady_clock;

void set_timing_enabled(bool enabled);
bool timing_enabled();
double timing_elapsed_ms(const TimingClock::time_point& start);

}  // namespace glim
