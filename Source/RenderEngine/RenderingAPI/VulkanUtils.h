#pragma once

#include <vulkan/vulkan.hpp>

class VulkanUtils
{
public:
    static std::vector<VkVertexInputBindingDescription> getVertexBindingDescriptions();
    static std::vector<VkVertexInputAttributeDescription> getVertexAttributeDescriptions();
};