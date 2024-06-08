#include "TestLogger.h"

#include <RenderEngine/RenderingAPI/Instance.h>

#include "Logs/LogSystem.h"

void TestLogger::log(LogSeverity severity, const char* message)
{
    if (isVulkanValidationLayersError(message))
    {
        ++number_of_vulkan_validation_layers_messages;
    }
}

bool TestLogger::isVulkanValidationLayersError(const std::string_view& message)
{
    return message.contains(Instance::VALIDATION_LAYERS_PREFIX) && message.contains(LogSystem::LOG_SEVERITY_PREFIXES.at(LogSeverity::ERROR));
}
