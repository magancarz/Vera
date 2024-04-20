#pragma once

#include "RenderEngine/RenderingAPI/Device.h"
#include "PushConstantRay.h"
#include "RenderEngine/RenderingAPI/Buffer.h"
#include "RenderEngine/RenderingAPI/ShaderModule.h"

class RayTracingPipeline
{
public:
    RayTracingPipeline(Device& device, VkDescriptorSetLayout descriptor_set_layout, const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& ray_tracing_properties);

    RayTracingPipeline(const RayTracingPipeline&) = delete;
    RayTracingPipeline& operator=(const RayTracingPipeline&) = delete;

    void bind(VkCommandBuffer command_buffer);
    void bindDescriptorSets(VkCommandBuffer command_buffer, const std::vector<VkDescriptorSet>& descriptor_sets);
    void pushConstants(VkCommandBuffer command_buffer, const PushConstantRay& push_constant_ray);

    VkStridedDeviceAddressRegionKHR rchitShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR rgenShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR rmissShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR callableShaderBindingTable{};

private:
    void createPipeline();
    void createShaderBindingTable();

    Device& device;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    std::unique_ptr<Buffer> shader_binding_table;

    VkPipelineLayout pipelineLayoutHandle{VK_NULL_HANDLE};
    VkPipeline rayTracingPipelineHandle{VK_NULL_HANDLE};

    VkDeviceAddress shaderBindingTableBufferDeviceAddress;
    VkDeviceSize hitGroupOffset;
    VkDeviceSize rayGenOffset;
    VkDeviceSize missOffset;
};
