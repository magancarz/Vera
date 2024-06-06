#include "RayTracingPipeline.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "RenderEngine/SceneRenderers/RayTraced/PushConstantRay.h"
#include "Memory/Buffer.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"

#include <fstream>

RayTracingPipeline::RayTracingPipeline(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        const std::vector<VkPipelineShaderStageCreateInfo>& shader_stage_create_info_list,
        const std::vector<VkRayTracingShaderGroupCreateInfoKHR>& shader_group_create_info_list,
        const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts,
        uint32_t max_recursion_depth,
        uint32_t miss_count,
        uint32_t hit_group_count,
        const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& ray_tracing_properties)
    : device{device}, memory_allocator{memory_allocator}, ray_tracing_properties{ray_tracing_properties}
{
    createPipeline(shader_stage_create_info_list, shader_group_create_info_list, descriptor_set_layouts, max_recursion_depth);
    createShaderBindingTable(miss_count, hit_group_count);
}

void RayTracingPipeline::createPipeline(
        const std::vector<VkPipelineShaderStageCreateInfo>& shader_stage_create_info_list,
        const std::vector<VkRayTracingShaderGroupCreateInfoKHR>& shader_group_create_info_list,
        const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts,
        uint32_t max_recursion_depth)
{
    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    push_constant_range.offset = 0;
    push_constant_range.size = sizeof(PushConstantRay);

    VkPipelineLayoutCreateInfo pipeline_layout_create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_create_info.setLayoutCount = (uint32_t)descriptor_set_layouts.size();
    pipeline_layout_create_info.pSetLayouts = descriptor_set_layouts.data();
    pipeline_layout_create_info.pushConstantRangeCount = 1;
    pipeline_layout_create_info.pPushConstantRanges = &push_constant_range;

    if (vkCreatePipelineLayout(device.getDeviceHandle(), &pipeline_layout_create_info, VulkanDefines::NO_CALLBACK, &pipeline_layout_handle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    VkRayTracingPipelineCreateInfoKHR ray_tracing_pipeline_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_tracing_pipeline_create_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    ray_tracing_pipeline_create_info.stageCount = static_cast<uint32_t>(shader_stage_create_info_list.size());
    ray_tracing_pipeline_create_info.pStages = shader_stage_create_info_list.data();
    ray_tracing_pipeline_create_info.groupCount = static_cast<uint32_t>(shader_group_create_info_list.size());
    ray_tracing_pipeline_create_info.pGroups = shader_group_create_info_list.data();
    ray_tracing_pipeline_create_info.maxPipelineRayRecursionDepth = max_recursion_depth;
    ray_tracing_pipeline_create_info.layout = pipeline_layout_handle;

    if (pvkCreateRayTracingPipelinesKHR(
            device.getDeviceHandle(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
            &ray_tracing_pipeline_create_info, VulkanDefines::NO_CALLBACK, &pipeline_handle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create ray tracing pipeline!");
    }
}

void RayTracingPipeline::createShaderBindingTable(uint32_t miss_count, uint32_t hit_group_count)
{
    assert(miss_count > 0 && "There must be at least one miss shader!");
    assert(hit_group_count > 0 && "There must be at least one hit group!");

    constexpr uint32_t ray_gen_count = 1;
    uint32_t handle_count = ray_gen_count + miss_count + hit_group_count;
    uint32_t handle_size = ray_tracing_properties.shaderGroupHandleSize;

    uint32_t handle_size_aligned = VulkanHelper::align_up(handle_size, ray_tracing_properties.shaderGroupHandleAlignment);

    ray_gen_shader_binding_table.stride = VulkanHelper::align_up(handle_size_aligned, ray_tracing_properties.shaderGroupBaseAlignment);
    ray_gen_shader_binding_table.size = ray_gen_shader_binding_table.stride;

    miss_shader_binding_table.stride = handle_size_aligned;
    miss_shader_binding_table.size = VulkanHelper::align_up(miss_count * handle_size_aligned, ray_tracing_properties.shaderGroupBaseAlignment);

    hit_shader_binding_table.stride = handle_size_aligned;
    hit_shader_binding_table.size = VulkanHelper::align_up(hit_group_count * handle_size_aligned, ray_tracing_properties.shaderGroupBaseAlignment);

    uint32_t dataSize = handle_count * handle_size;
    std::vector<uint8_t> handles(dataSize);
    if (pvkGetRayTracingShaderGroupHandlesKHR(device.getDeviceHandle(), pipeline_handle, 0, handle_count, dataSize, handles.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Cannot create ray tracing shader group handles!");
    }

    VkDeviceSize sbt_size = ray_gen_shader_binding_table.size + miss_shader_binding_table.size + hit_shader_binding_table.size + callable_shader_binding_table.size;
    shader_binding_table = memory_allocator.createBuffer
    (
            sbt_size,
            1,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, shader_binding_table->getBuffer()};
    VkDeviceAddress sbt_address = vkGetBufferDeviceAddress(device.getDeviceHandle(), &info);
    ray_gen_shader_binding_table.deviceAddress = sbt_address;
    miss_shader_binding_table.deviceAddress = sbt_address + ray_gen_shader_binding_table.size;
    hit_shader_binding_table.deviceAddress = sbt_address + ray_gen_shader_binding_table.size + miss_shader_binding_table.size;

    auto get_handle = [&] (uint32_t i) { return handles.data() + i * handle_size; };
    shader_binding_table->map();
    auto sbt_buffer = reinterpret_cast<uint8_t*>(shader_binding_table->getMappedMemory());
    uint8_t* data;
    uint32_t handle_idx{0};

    data = sbt_buffer;
    memcpy(data, get_handle(handle_idx++), handle_size);

    data = sbt_buffer + ray_gen_shader_binding_table.size;
    for(uint32_t c = 0; c < miss_count; ++c)
    {
        memcpy(data, get_handle(handle_idx++), handle_size);
        data += miss_shader_binding_table.stride;
    }

    data = sbt_buffer + ray_gen_shader_binding_table.size + miss_shader_binding_table.size;
    for(uint32_t c = 0; c < hit_group_count; ++c)
    {
        memcpy(data, get_handle(handle_idx++), handle_size);
        data += hit_shader_binding_table.stride;
    }
    shader_binding_table->unmap();
}

RayTracingPipeline::~RayTracingPipeline()
{
    vkDestroyPipeline(device.getDeviceHandle(), pipeline_handle, VulkanDefines::NO_CALLBACK);
    vkDestroyPipelineLayout(device.getDeviceHandle(), pipeline_layout_handle, VulkanDefines::NO_CALLBACK);
}

void RayTracingPipeline::bind(VkCommandBuffer command_buffer)
{
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_handle);
}

void RayTracingPipeline::bindDescriptorSets(VkCommandBuffer command_buffer, const std::vector<VkDescriptorSet>& descriptor_sets)
{
    vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            pipeline_layout_handle,
            0,
            (uint32_t)descriptor_sets.size(),
            descriptor_sets.data(),
            0,
            nullptr);
}

void RayTracingPipeline::pushConstants(VkCommandBuffer command_buffer, const PushConstantRay& push_constant_ray)
{
    vkCmdPushConstants(
            command_buffer,
            pipeline_layout_handle,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
            0,
            sizeof(PushConstantRay),
            &push_constant_ray);
}

ShaderBindingTableValues RayTracingPipeline::getShaderBindingTableValues()
{
    return
    {
            .ray_gen_shader_binding_table = ray_gen_shader_binding_table,
            .miss_shader_binding_table = miss_shader_binding_table,
            .closest_hit_shader_binding_table = hit_shader_binding_table,
            .callable_shader_binding_table = callable_shader_binding_table
    };
}