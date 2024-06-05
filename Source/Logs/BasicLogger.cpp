#include "BasicLogger.h"

#include <iostream>

void BasicLogger::log(LogSeverity severity, const char* message)
{
    if (severity <= LogSeverity::WARNING)
    {
        std::cout << message;
        return;
    }

    std::cerr << message;
}
