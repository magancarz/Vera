#include "RayTracingPipelineBuilder.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

RayTracingPipelineBuilder::RayTracingPipelineBuilder(Device& device, VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties)
    : device{device}, ray_tracing_properties{ray_tracing_properties} {}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addRayGenerationStage(std::shared_ptr<ShaderModule> ray_gen)
{
    assert(ray_gen_count < 1 && "There can be only one ray gen shader!");
    ++ray_gen_count;
    shader_modules.emplace_back(ray_gen);

    uint32_t ray_gen_shader_index = addShaderStage(ray_gen, VK_SHADER_STAGE_RAYGEN_BIT_KHR);

    VkRayTracingShaderGroupCreateInfoKHR ray_generate_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    ray_generate_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    ray_generate_shader_group_create_info.generalShader = ray_gen_shader_index;
    ray_generate_shader_group_create_info.closestHitShader = VK_SHADER_UNUSED_KHR;
    ray_generate_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    ray_generate_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_group_create_info_list.push_back(ray_generate_shader_group_create_info);

    printf("Placing ray gen shader at %zu index\n", shader_group_create_info_list.size() - 1);

    return *this;
}

uint32_t RayTracingPipelineBuilder::addShaderStage(
        const std::shared_ptr<ShaderModule>& shader_module,
        VkShaderStageFlagBits shader_stage)
{
    uint32_t shader_stage_index = shader_stage_create_info_list.size();
    shader_stage_create_info_list.push_back(createShaderStageCreateInfo(shader_module, shader_stage));
    return shader_stage_index;
}

VkPipelineShaderStageCreateInfo RayTracingPipelineBuilder::createShaderStageCreateInfo(
        const std::shared_ptr<ShaderModule>& shader_module,
        VkShaderStageFlagBits shader_stage)
{
    VkPipelineShaderStageCreateInfo stage_create_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_create_info.pName = "main";
    stage_create_info.stage = shader_stage;
    stage_create_info.module = shader_module->getShaderModule();

    return stage_create_info;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addMissStage(std::shared_ptr<ShaderModule> miss)
{
    shader_modules.emplace_back(miss);
    ++miss_count;

    uint32_t miss_shader_index = addShaderStage(miss, VK_SHADER_STAGE_MISS_BIT_KHR);

    VkRayTracingShaderGroupCreateInfoKHR miss_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    miss_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    miss_shader_group_create_info.generalShader = miss_shader_index;
    miss_shader_group_create_info.closestHitShader = VK_SHADER_UNUSED_KHR;
    miss_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    miss_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_group_create_info_list.push_back(miss_shader_group_create_info);

    printf("Placing miss shader at %zu index\n", shader_group_create_info_list.size() - 1);

    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addHitGroup(
        std::shared_ptr<ShaderModule> hit,
        std::shared_ptr<ShaderModule> any_hit)
{
    ++hit_group_count;
    shader_modules.emplace_back(hit);
    shader_modules.emplace_back(any_hit);

    uint32_t hit_shader_index = addShaderStage(hit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    uint32_t any_hit_shader_index = addShaderStage(any_hit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

    VkRayTracingShaderGroupCreateInfoKHR hit_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    hit_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    hit_shader_group_create_info.closestHitShader = hit_shader_index;
    hit_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
    hit_shader_group_create_info.anyHitShader = any_hit_shader_index;
    hit_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_group_create_info_list.push_back(hit_shader_group_create_info);

    printf("Placing hit group at %zu index\n", shader_group_create_info_list.size() - 1);

    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addHitGroupWithOnlyHitShader(std::shared_ptr<ShaderModule> hit)
{
    ++hit_group_count;
    shader_modules.emplace_back(hit);

    uint32_t hit_shader_index = addShaderStage(hit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

    VkRayTracingShaderGroupCreateInfoKHR closest_hit_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    closest_hit_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    closest_hit_shader_group_create_info.closestHitShader = hit_shader_index;
    closest_hit_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
    closest_hit_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    closest_hit_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_group_create_info_list.push_back(closest_hit_shader_group_create_info);

    printf("Placing hit group with only hit shader at %zu index\n", shader_group_create_info_list.size() - 1);

    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addHitGroupWithOnlyAnyHitShader(std::shared_ptr<ShaderModule> hit)
{
    ++hit_group_count;
    shader_modules.emplace_back(hit);

    uint32_t any_hit_shader_index = addShaderStage(hit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

    VkRayTracingShaderGroupCreateInfoKHR any_hit_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    any_hit_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    any_hit_shader_group_create_info.closestHitShader = VK_SHADER_UNUSED_KHR;
    any_hit_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
    any_hit_shader_group_create_info.anyHitShader = any_hit_shader_index;
    any_hit_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_group_create_info_list.push_back(any_hit_shader_group_create_info);

    printf("Placing hit group with only any hit shader at %zu index\n", shader_group_create_info_list.size() - 1);

    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::setMaxRecursionDepth(uint32_t max_recursion_depth)
{
    assert(max_recursion_depth >= 1 && "Max recursion depth must be at least greater or equal to 1!");
    this->max_recursion = max_recursion_depth;

    printf("Setting ray tracing pipeline max recursion depth to %d\n", max_recursion_depth);

    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addDescriptorSetLayout(VkDescriptorSetLayout descriptor_set_layout)
{
    descriptor_set_layouts.emplace_back(descriptor_set_layout);

    return *this;
}

std::unique_ptr<RayTracingPipeline> RayTracingPipelineBuilder::build()
{
    return std::make_unique<RayTracingPipeline>(
            device,
            shader_stage_create_info_list,
            shader_group_create_info_list,
            descriptor_set_layouts,
            max_recursion,
            miss_count,
            hit_group_count,
            ray_tracing_properties
            );
}