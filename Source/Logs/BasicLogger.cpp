#include "BasicLogger.h"

#include <iostream>

void BasicLogger::log(LogSeverity severity, const char* message)
{
    if (severity <= LogSeverity::WARNING)
    {
        std::cout << LOG_SEVERITY_PREFIXES.at(severity) << message << '\n';
        return;
    }

    std::cerr << LOG_SEVERITY_PREFIXES.at(severity) << message << '\n';
}
