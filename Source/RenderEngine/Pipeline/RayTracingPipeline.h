#pragma once

#include "RenderEngine/RenderingAPI/VulkanHandler.h"
#include "RenderEngine/SceneRenderers/RayTraced/PushConstantRay.h"
#include "Memory/Buffer.h"
#include "RenderEngine/RenderingAPI/ShaderModule.h"
#include "Memory/MemoryAllocator.h"

struct ShaderBindingTableValues
{
    VkStridedDeviceAddressRegionKHR ray_gen_shader_binding_table{};
    VkStridedDeviceAddressRegionKHR miss_shader_binding_table{};
    VkStridedDeviceAddressRegionKHR closest_hit_shader_binding_table{};
    VkStridedDeviceAddressRegionKHR callable_shader_binding_table{};
};

class RayTracingPipeline
{
public:
    RayTracingPipeline(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        const std::vector<VkPipelineShaderStageCreateInfo>& shader_stage_create_info_list,
        const std::vector<VkRayTracingShaderGroupCreateInfoKHR>& shader_group_create_info_list,
        const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts,
        uint32_t max_recursion_depth,
        uint32_t miss_count,
        uint32_t hit_group_count,
        const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& ray_tracing_properties);
    ~RayTracingPipeline();

    RayTracingPipeline(const RayTracingPipeline&) = delete;
    RayTracingPipeline& operator=(const RayTracingPipeline&) = delete;

    void bind(VkCommandBuffer command_buffer);
    void bindDescriptorSets(VkCommandBuffer command_buffer, const std::vector<VkDescriptorSet>& descriptor_sets);
    void pushConstants(VkCommandBuffer command_buffer, const PushConstantRay& push_constant_ray);

    ShaderBindingTableValues getShaderBindingTableValues();

private:
    void createPipeline(
        const std::vector<VkPipelineShaderStageCreateInfo>& shader_stage_create_info_list,
        const std::vector<VkRayTracingShaderGroupCreateInfoKHR>& shader_group_create_info_list,
        const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts,
        uint32_t max_recursion_depth);
    void createShaderBindingTable(uint32_t miss_count, uint32_t hit_group_count);

    VulkanHandler& device;
    MemoryAllocator& memory_allocator;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    VkPipelineLayout pipeline_layout_handle{VK_NULL_HANDLE};
    VkPipeline pipeline_handle{VK_NULL_HANDLE};

    std::unique_ptr<Buffer> shader_binding_table;
    VkStridedDeviceAddressRegionKHR hit_shader_binding_table{};
    VkStridedDeviceAddressRegionKHR ray_gen_shader_binding_table{};
    VkStridedDeviceAddressRegionKHR miss_shader_binding_table{};
    VkStridedDeviceAddressRegionKHR callable_shader_binding_table{};
};
