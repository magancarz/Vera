#pragma once

#include <chrono>

namespace utils
{
    class Timer
    {
    public:
        Timer();

        double getTimeInMillis() const;

    private:
        std::chrono::time_point<std::chrono::steady_clock> start;
    };
}
