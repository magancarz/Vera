#include "TestLogger.h"

#include <RenderEngine/RenderingAPI/Instance.h>

void TestLogger::log(LogSeverity severity, const char* message)
{
    if (isVulkanValidationLayersMessage(message))
    {
        ++number_of_vulkan_validation_layers_messages;
    }
}

bool TestLogger::isVulkanValidationLayersMessage(const std::string& message)
{
    return message.contains(Instance::VALIDATION_LAYERS_PREFIX);
}
