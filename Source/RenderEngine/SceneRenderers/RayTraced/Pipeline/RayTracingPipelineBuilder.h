#pragma once

#include "RayTracingPipeline.h"

class RayTracingPipelineBuilder
{
public:
    RayTracingPipelineBuilder(Device& device, VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties);

    RayTracingPipelineBuilder& addRayGenerationStage(std::shared_ptr<ShaderModule> ray_gen);
    RayTracingPipelineBuilder& addMissStage(std::shared_ptr<ShaderModule> miss);
    RayTracingPipelineBuilder& addHitGroupWithOnlyHitShader(std::shared_ptr<ShaderModule> hit);
    RayTracingPipelineBuilder& addHitGroupWithOnlyAnyHitShader(std::shared_ptr<ShaderModule> any_hit);

    RayTracingPipelineBuilder& setMaxRecursionDepth(uint32_t max_recursion_depth);
    RayTracingPipelineBuilder& addDescriptorSetLayout(VkDescriptorSetLayout descriptor_set_layout);

    std::unique_ptr<RayTracingPipeline> build();

private:
    Device& device;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    static VkPipelineShaderStageCreateInfo createShaderStageCreateInfo(
            const std::shared_ptr<ShaderModule>& shader_module,
            VkShaderStageFlagBits shader_stage);

    std::vector<std::shared_ptr<ShaderModule>> shader_modules;
    std::vector<VkPipelineShaderStageCreateInfo> shader_stage_create_info_list;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_group_create_info_list;

    uint32_t ray_gen_count{0};
    uint32_t miss_count{0};
    uint32_t hit_group_count{0};

    uint32_t max_recursion{1};
    std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
};
