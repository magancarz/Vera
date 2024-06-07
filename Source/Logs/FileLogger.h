#pragma once

#include <memory>
#include <fstream>

#include "Logger.h"
#include "LoggerDecorator.h"

class FileLogger : public LoggerDecorator
{
public:
    explicit FileLogger(std::string log_file_location);
    FileLogger(std::unique_ptr<Logger> next_logger, std::string log_file_location);
    ~FileLogger() override;

    FileLogger(const FileLogger&) = delete;
    FileLogger& operator=(const FileLogger&) = delete;

    void log(LogSeverity severity, const char* message) override;

private:
    std::ofstream log_file;
    std::string log_file_location;
};
