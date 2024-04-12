#pragma once

#include "RenderEngine/RenderingAPI/Device.h"

class RayTracingPipeline
{
public:
    RayTracingPipeline(Device& device, VkDescriptorSetLayout descriptor_set_layout, VkDescriptorSetLayout material_descriptor_set_layout, const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& ray_tracing_properties);

    RayTracingPipeline(const RayTracingPipeline&) = delete;
    RayTracingPipeline& operator=(const RayTracingPipeline&) = delete;

    void bind(VkCommandBuffer command_buffer);
    void bindDescriptorSets(VkCommandBuffer command_buffer, const std::vector<VkDescriptorSet>& descriptor_sets);

    VkStridedDeviceAddressRegionKHR rchitShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR rgenShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR rmissShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR callableShaderBindingTable{};

private:
    void createPipeline();

    Device& device;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSetLayout material_descriptor_set_layout;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    VkPipelineLayout pipelineLayoutHandle{VK_NULL_HANDLE};
    VkPipeline rayTracingPipelineHandle{VK_NULL_HANDLE};

    VkDeviceAddress shaderBindingTableBufferDeviceAddress;
    VkDeviceSize hitGroupOffset;
    VkDeviceSize rayGenOffset;
    VkDeviceSize missOffset;
};
