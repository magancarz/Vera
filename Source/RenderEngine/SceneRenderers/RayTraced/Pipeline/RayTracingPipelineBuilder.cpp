#include "RayTracingPipelineBuilder.h"

#include "Logs/LogSystem.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

RayTracingPipelineBuilder::RayTracingPipelineBuilder(
        VulkanFacade& device,
        std::unique_ptr<MemoryAllocator>& memory_allocator,
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties)
    : device{device}, memory_allocator{memory_allocator}, ray_tracing_properties{ray_tracing_properties} {}

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

    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addDefaultOcclusionCheckShader(std::shared_ptr<ShaderModule> occlusion_shader)
{
    default_occlusion_shader_stage_index = addAnyHitStage(std::move(occlusion_shader));
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addMaterialShader(const std::string& material_name, std::shared_ptr<ShaderModule> hit)
{
    if (material_shaders.contains(material_name))
    {
        return *this;
    }

    uint32_t closest_hit_stage_index = addClosestHitStage(std::move(hit));
    material_shaders[material_name] = MaterialShader
    {
            .closest_hit_shader_stage_index = closest_hit_stage_index
    };

    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addMaterialShader(
        const std::string& material_name,
        std::shared_ptr<ShaderModule> hit,
        std::shared_ptr<ShaderModule> any_hit)
{
    if (material_shaders.contains(material_name))
    {
        return *this;
    }

    uint32_t closest_hit_stage_index = addClosestHitStage(std::move(hit));
    uint32_t any_hit_stage_index = addAnyHitStage(std::move(any_hit));
    material_shaders[material_name] = MaterialShader
    {
        .closest_hit_shader_stage_index = closest_hit_stage_index,
        .any_hit_shader_stage_index = any_hit_stage_index
    };

    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addMaterialShader(
        const std::string& material_name,
        std::shared_ptr<ShaderModule> hit,
        std::shared_ptr<ShaderModule> any_hit,
        std::shared_ptr<ShaderModule> occlusion)
{
    if (material_shaders.contains(material_name))
    {
        return *this;
    }

    uint32_t closest_hit_stage_index = addClosestHitStage(std::move(hit));
    uint32_t any_hit_stage_index = addAnyHitStage(std::move(any_hit));
    uint32_t occlusion_stage_index = addAnyHitStage(std::move(occlusion));
    material_shaders[material_name] = MaterialShader
    {
            .closest_hit_shader_stage_index = closest_hit_stage_index,
            .any_hit_shader_stage_index = any_hit_stage_index,
            .occlusion_shader_stage_index = occlusion_stage_index
    };

    return *this;
}

uint32_t RayTracingPipelineBuilder::addClosestHitStage(std::shared_ptr<ShaderModule> hit)
{
    uint32_t hit_shader_index = addShaderStage(hit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    shader_modules.emplace_back(std::move(hit));
    return hit_shader_index;
}

uint32_t RayTracingPipelineBuilder::addAnyHitStage(std::shared_ptr<ShaderModule> any_hit)
{
    uint32_t any_hit_shader_index = addShaderStage(any_hit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
    shader_modules.emplace_back(std::move(any_hit));
    return any_hit_shader_index;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::registerObjectMaterial(const std::string& material_name)
{
    MaterialShader material_shader = material_shaders[material_name];
    if (material_shader.any_hit_shader_stage_index.has_value())
    {
        addHitGroup(material_shader.closest_hit_shader_stage_index, material_shader.any_hit_shader_stage_index.value());
    } else
    {
        addHitGroup(material_shader.closest_hit_shader_stage_index);
    }

    assert(default_occlusion_shader_stage_index.has_value() && "There must be at least default occlusion shader stage");
    if (material_shader.occlusion_shader_stage_index.has_value())
    {
        addOcclusionCheckGroup(material_shader.occlusion_shader_stage_index.value());
    }
    else
    {
        addOcclusionCheckGroup(default_occlusion_shader_stage_index.value());
    }

    return *this;
}

void RayTracingPipelineBuilder::addHitGroup(uint32_t closest_hit_stage_index)
{
    ++hit_group_count;
    VkRayTracingShaderGroupCreateInfoKHR closest_hit_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    closest_hit_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    closest_hit_shader_group_create_info.closestHitShader = closest_hit_stage_index;
    closest_hit_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
    closest_hit_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    closest_hit_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_group_create_info_list.push_back(closest_hit_shader_group_create_info);
}

void RayTracingPipelineBuilder::addHitGroup(uint32_t closest_hit_stage_index, uint32_t any_hit_stage_index)
{
    ++hit_group_count;
    VkRayTracingShaderGroupCreateInfoKHR closest_hit_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    closest_hit_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    closest_hit_shader_group_create_info.closestHitShader = closest_hit_stage_index;
    closest_hit_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
    closest_hit_shader_group_create_info.anyHitShader = any_hit_stage_index;
    closest_hit_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_group_create_info_list.push_back(closest_hit_shader_group_create_info);
}

void RayTracingPipelineBuilder::addOcclusionCheckGroup(uint32_t any_hit_stage_index)
{
    ++hit_group_count;
    VkRayTracingShaderGroupCreateInfoKHR closest_hit_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    closest_hit_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    closest_hit_shader_group_create_info.closestHitShader = VK_SHADER_UNUSED_KHR;
    closest_hit_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
    closest_hit_shader_group_create_info.anyHitShader = any_hit_stage_index;
    closest_hit_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_group_create_info_list.push_back(closest_hit_shader_group_create_info);
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::setMaxRecursionDepth(uint32_t max_recursion_depth)
{
    assert(max_recursion_depth >= 1 && "Max recursion depth must be at least greater or equal to 1!");
    max_recursion = max_recursion_depth;

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
            memory_allocator,
            shader_stage_create_info_list,
            shader_group_create_info_list,
            descriptor_set_layouts,
            max_recursion,
            miss_count,
            hit_group_count,
            ray_tracing_properties
            );
}
