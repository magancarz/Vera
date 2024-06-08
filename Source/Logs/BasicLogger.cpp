#include "BasicLogger.h"

#include <iostream>

void BasicLogger::log(LogSeverity severity, const char* message)
{
    switch (severity)
    {
    case LogSeverity::ERROR:
        std::cerr << message;
        break;
    case LogSeverity::WARNING:
        std::cout << "\033[0;33m" << message << "\033[0m";
        break;
    default:
        std::cout << message;
        break;
    }
}
