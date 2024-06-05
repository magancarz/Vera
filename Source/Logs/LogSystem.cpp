#include "LogSystem.h"

#include <chrono>
#include <cstring>
#include <ctime>

void LogSystem::initialize(std::unique_ptr<Logger> logger)
{
    app_logger = std::move(logger);
}

std::string LogSystem::getCurrentTimestamp()
{
    std::time_t result = std::time(nullptr);
    char* time_str = std::asctime(std::localtime(&result));
    time_str[strlen(time_str) - 1] = '\0';
    std::ostringstream oss;
    oss << "[" << time_str << "] ";

    return oss.str();
}
