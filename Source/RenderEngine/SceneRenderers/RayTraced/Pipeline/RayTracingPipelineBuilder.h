#pragma once

#include "RayTracingPipeline.h"
#include "RenderEngine/Materials/Material.h"
#include "MaterialShader.h"

class RayTracingPipelineBuilder
{
public:
    RayTracingPipelineBuilder(
            VulkanFacade& device,
            MemoryAllocator& memory_allocator,
            VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties);

    RayTracingPipelineBuilder& addRayGenerationStage(std::shared_ptr<ShaderModule> ray_gen);
    RayTracingPipelineBuilder& addMissStage(std::shared_ptr<ShaderModule> miss);
    RayTracingPipelineBuilder& addDefaultOcclusionCheckShader(std::shared_ptr<ShaderModule> occlusion_shader);
    RayTracingPipelineBuilder& addMaterialShader(
            const std::string& material_name,
            std::shared_ptr<ShaderModule> hit);
    RayTracingPipelineBuilder& addMaterialShader(
            const std::string& material_name,
            std::shared_ptr<ShaderModule> hit,
            std::shared_ptr<ShaderModule> any_hit);
    RayTracingPipelineBuilder& addMaterialShader(
            const std::string& material_name,
            std::shared_ptr<ShaderModule> hit,
            std::shared_ptr<ShaderModule> any_hit,
            std::shared_ptr<ShaderModule> occlusion);
    RayTracingPipelineBuilder& registerObjectMaterial(
            const std::string& material_name);
    RayTracingPipelineBuilder& setMaxRecursionDepth(uint32_t max_recursion_depth);
    RayTracingPipelineBuilder& addDescriptorSetLayout(VkDescriptorSetLayout descriptor_set_layout);

    std::unique_ptr<RayTracingPipeline> build();

private:
    VulkanFacade& device;
    MemoryAllocator& memory_allocator;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    uint32_t addClosestHitStage(std::shared_ptr<ShaderModule> hit);
    uint32_t addAnyHitStage(std::shared_ptr<ShaderModule> any_hit);
    uint32_t addShaderStage(
            const std::shared_ptr<ShaderModule>& shader_module,
            VkShaderStageFlagBits shader_stage);
    static VkPipelineShaderStageCreateInfo createShaderStageCreateInfo(
            const std::shared_ptr<ShaderModule>& shader_module,
            VkShaderStageFlagBits shader_stage);
    void addHitGroup(uint32_t closest_hit_stage_index);
    void addHitGroup(uint32_t closest_hit_stage_index, uint32_t any_hit_stage_index);
    void addOcclusionCheckGroup(uint32_t any_hit_stage_index);

    std::unordered_map<std::string, MaterialShader> material_shaders;

    std::vector<std::shared_ptr<ShaderModule>> shader_modules;

    std::optional<uint32_t> default_occlusion_shader_stage_index;
    std::vector<VkPipelineShaderStageCreateInfo> shader_stage_create_info_list;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_group_create_info_list;

    uint32_t ray_gen_count{0};
    uint32_t miss_count{0};
    uint32_t hit_group_count{0};

    uint32_t max_recursion{1};
    std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
};
