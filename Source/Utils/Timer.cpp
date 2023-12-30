#include "Timer.h"

#include <chrono>

utils::Timer::Timer()
{
    start = std::chrono::steady_clock::now();
}

double utils::Timer::getTimeInMillis() const
{
    const auto end = std::chrono::steady_clock::now();
    return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}
