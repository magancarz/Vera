#include "VulkanUtils.h"

#include "Assets/Model/Vertex.h"
#include "RenderEngine/RenderingAPI/PhysicalDevice.h"

std::vector<VkVertexInputBindingDescription> VulkanUtils::getVertexBindingDescriptions()
{
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return {binding};
}

std::vector<VkVertexInputAttributeDescription> VulkanUtils::getVertexAttributeDescriptions()
{
    std::vector<VkVertexInputAttributeDescription> attribute_description{};
    attribute_description.push_back({.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = static_cast<uint32_t>(offsetof(Vertex, position))});
    attribute_description.push_back({.location = 1, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = static_cast<uint32_t>(offsetof(Vertex, uv))});

    return attribute_description;
}

VkPhysicalDeviceRayTracingPipelinePropertiesKHR VulkanUtils::queryRayTracingPipelineProperties(PhysicalDevice& physical_device)
{
    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties{};
    ray_tracing_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2 physical_device_properties_2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    physical_device_properties_2.pNext = &ray_tracing_properties;
    physical_device_properties_2.properties = physical_device_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &physical_device_properties_2);

    return ray_tracing_properties;
}