#pragma once

#include <unordered_map>
#include <string>

#include "Logger.h"

class BasicLogger : public Logger
{
public:
    void log(LogSeverity severity, const char* message) override;

    inline static std::unordered_map<LogSeverity, std::string> LOG_SEVERITY_PREFIXES{
        {LogSeverity::LOG, "Log: "},
        {LogSeverity::WARNING, "Warning: "},
        {LogSeverity::ERROR, "Error: "},
        {LogSeverity::FATAL, "Fatal: "}
    };
};
