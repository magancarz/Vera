#pragma once

#include <memory>

#include "LogSeverity.h"

class Logger;

class LogSystem
{
public:
    static void initialize(std::unique_ptr<Logger> logger);
    static void log(LogSeverity severity, const char* message);

private:
    inline static std::unique_ptr<Logger> app_logger;
};
