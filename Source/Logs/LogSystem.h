#pragma once

#include <cassert>
#include <memory>
#include <sstream>

#include "Logger.h"
#include "LogSeverity.h"

class LogSystem
{
public:
    static void initialize(std::unique_ptr<Logger> logger);

    template<typename... Args>
    static void log(LogSeverity severity, Args... args)
    {
        assert(app_logger && "Cannot log if application logger is not properly initialized!");
        std::ostringstream oss;
        format(oss, args...);
        std::string message = oss.str();
        app_logger->log(severity, message.c_str());
    }

private:
    template<typename T>
    static void format(std::ostringstream& oss, T t)
    {
        oss << t;
    }

    template<typename T, typename... Args>
    static void format(std::ostringstream& oss, T t, Args... args)
    {
        oss << t;
        format(oss, args...);
    }

    inline static std::unique_ptr<Logger> app_logger;
};
