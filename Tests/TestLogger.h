#pragma once

#include <cstdint>
#include <string>

#include <Logs/Logger.h>

class TestLogger : public Logger
{
public:
    void log(LogSeverity severity, const char* message) override;

    [[nodiscard]] bool anyVulkanValidationLayersErrorsOrWarnings() const { return number_of_vulkan_validation_layers_messages > 0; }

private:
    bool isVulkanValidationLayersMessage(const std::string& message);

    uint32_t number_of_vulkan_validation_layers_messages{0};
};
