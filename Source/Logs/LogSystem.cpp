#include "LogSystem.h"

#include <cassert>

#include "Logger.h"

void LogSystem::initialize(std::unique_ptr<Logger> logger)
{
    app_logger = std::move(logger);
}

void LogSystem::log(LogSeverity severity, const char* message)
{
    assert(app_logger && "Cannot log if application logger is not properly initialized!");
    app_logger->log(severity, message);
}
