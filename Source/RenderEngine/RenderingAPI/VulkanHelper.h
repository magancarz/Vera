#pragma once

#include <cstdint>
#include <vulkan/vulkan.hpp>

class VulkanHelper
{
public:
    static bool checkResult(VkResult result, const char* message = nullptr);
    static bool checkResult(VkResult result, const char* file, int32_t line);
    static const char* getResultString(VkResult result);
};
