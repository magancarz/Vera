#include "RayTracingPipeline.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "PushConstantRay.h"
#include "RenderEngine/RenderingAPI/Buffer.h"

#include <fstream>

RayTracingPipeline::RayTracingPipeline(Device& device, VkDescriptorSetLayout descriptor_set_layout,
       const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& ray_tracing_properties)
    : device{device}, descriptor_set_layout{descriptor_set_layout},
      ray_tracing_properties{ray_tracing_properties}
{
    createPipeline();
    createShaderBindingTable();
}

void RayTracingPipeline::createPipeline()
{
    std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandleList = {descriptor_set_layout};

    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    push_constant_range.offset = 0;
    push_constant_range.size = sizeof(PushConstantRay);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .setLayoutCount = (uint32_t)descriptorSetLayoutHandleList.size(),
            .pSetLayouts = descriptorSetLayoutHandleList.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range};

    if (vkCreatePipelineLayout(device.getDevice(), &pipelineLayoutCreateInfo, NULL, &pipelineLayoutHandle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    // =========================================================================
    // Ray Tracing Pipeline

    enum StageIndices
    {
        ray_gen,
        miss,
        miss_shadow,
        closest_hit_lambertian,
        any_hit_occlusion,
//        closest_hit_light,
//        closest_hit_specular,
//        closest_hit_dielectric,
        shader_group_count
    };

    std::vector<VkPipelineShaderStageCreateInfo> pipeline_shader_stage_create_info_list;
    pipeline_shader_stage_create_info_list.reserve(shader_group_count);
    VkPipelineShaderStageCreateInfo stage_create_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_create_info.pName = "main";

    ShaderModule ray_gen_shader_module{device, "Shaders/raytrace.rgen.spv"};
    stage_create_info.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stage_create_info.module = ray_gen_shader_module.getShaderModule();
    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

//    ShaderModule closest_hit_light_shader_module{device, "Shaders/raytrace_light.rchit.spv"};
//    stage_create_info.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
//    stage_create_info.module = closest_hit_light_shader_module.getShaderModule();
//    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

//    ShaderModule closest_hit_specular_shader_module{device, "Shaders/raytrace_specular.rchit.spv"};
//    stage_create_info.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
//    stage_create_info.module = closest_hit_specular_shader_module.getShaderModule();
//    pipeline_shader_stage_create_info_list.push_back(stage_create_info);
//
//    ShaderModule closest_hit_dielectric_shader_module{device, "Shaders/raytrace_dielectric.rchit.spv"};
//    stage_create_info.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
//    stage_create_info.module = closest_hit_dielectric_shader_module.getShaderModule();
//    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

    ShaderModule miss_shader_module{device, "Shaders/raytrace.rmiss.spv"};
    stage_create_info.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stage_create_info.module = miss_shader_module.getShaderModule();
    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

    ShaderModule miss_shadow_shader_module{device, "Shaders/raytrace_shadow.rmiss.spv"};
    stage_create_info.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stage_create_info.module = miss_shadow_shader_module.getShaderModule();
    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

    ShaderModule closest_hit_lambertian_shader_module{device, "Shaders/raytrace_lambertian.rchit.spv"};
    stage_create_info.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stage_create_info.module = closest_hit_lambertian_shader_module.getShaderModule();
    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

    ShaderModule any_hit_occlusion_shader_module{device, "Shaders/raytrace_occlusion.rchit.spv"};
    stage_create_info.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stage_create_info.module = any_hit_occlusion_shader_module.getShaderModule();
    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> ray_tracing_shader_group_create_info_list;

    VkRayTracingShaderGroupCreateInfoKHR ray_generate_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    ray_generate_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    ray_generate_shader_group_create_info.generalShader = ray_gen;
    ray_generate_shader_group_create_info.closestHitShader = VK_SHADER_UNUSED_KHR;
    ray_generate_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    ray_generate_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    ray_tracing_shader_group_create_info_list.push_back(ray_generate_shader_group_create_info);

//    VkRayTracingShaderGroupCreateInfoKHR closest_hit_light_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
//    closest_hit_light_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
//    closest_hit_light_shader_group_create_info.closestHitShader = closest_hit_light;
//    closest_hit_light_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
//    closest_hit_light_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
//    closest_hit_light_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
//    ray_tracing_shader_group_create_info_list.push_back(closest_hit_light_shader_group_create_info);
//
//    VkRayTracingShaderGroupCreateInfoKHR closest_hit_specular_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
//    closest_hit_specular_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
//    closest_hit_specular_shader_group_create_info.closestHitShader = closest_hit_specular;
//    closest_hit_specular_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
//    closest_hit_specular_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
//    closest_hit_specular_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
//    ray_tracing_shader_group_create_info_list.push_back(closest_hit_specular_shader_group_create_info);
//
//    VkRayTracingShaderGroupCreateInfoKHR closest_hit_dielectric_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
//    closest_hit_dielectric_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
//    closest_hit_dielectric_shader_group_create_info.closestHitShader = closest_hit_dielectric;
//    closest_hit_dielectric_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
//    closest_hit_dielectric_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
//    closest_hit_dielectric_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
//    ray_tracing_shader_group_create_info_list.push_back(closest_hit_dielectric_shader_group_create_info);

    VkRayTracingShaderGroupCreateInfoKHR miss_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    miss_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    miss_shader_group_create_info.generalShader = miss;
    miss_shader_group_create_info.closestHitShader = VK_SHADER_UNUSED_KHR;
    miss_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    miss_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    ray_tracing_shader_group_create_info_list.push_back(miss_shader_group_create_info);

    VkRayTracingShaderGroupCreateInfoKHR miss_shadow_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    miss_shadow_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    miss_shadow_shader_group_create_info.generalShader = miss_shadow;
    miss_shadow_shader_group_create_info.closestHitShader = VK_SHADER_UNUSED_KHR;
    miss_shadow_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    miss_shadow_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    ray_tracing_shader_group_create_info_list.push_back(miss_shadow_shader_group_create_info);

    VkRayTracingShaderGroupCreateInfoKHR closest_hit_lambertian_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    closest_hit_lambertian_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    closest_hit_lambertian_shader_group_create_info.closestHitShader = closest_hit_lambertian;
    closest_hit_lambertian_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
    closest_hit_lambertian_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    closest_hit_lambertian_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    ray_tracing_shader_group_create_info_list.push_back(closest_hit_lambertian_shader_group_create_info);

    VkRayTracingShaderGroupCreateInfoKHR any_hit_occlusion_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    any_hit_occlusion_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    any_hit_occlusion_shader_group_create_info.closestHitShader = any_hit_occlusion;
    any_hit_occlusion_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
    any_hit_occlusion_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    any_hit_occlusion_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    ray_tracing_shader_group_create_info_list.push_back(any_hit_occlusion_shader_group_create_info);

    VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            .pNext = NULL,
            .flags = 0,
            .stageCount = static_cast<uint32_t>(pipeline_shader_stage_create_info_list.size()),
            .pStages = pipeline_shader_stage_create_info_list.data(),
            .groupCount = static_cast<uint32_t>(ray_tracing_shader_group_create_info_list.size()),
            .pGroups = ray_tracing_shader_group_create_info_list.data(),
            .maxPipelineRayRecursionDepth = 2,
            .pLibraryInfo = NULL,
            .pLibraryInterface = NULL,
            .pDynamicState = NULL,
            .layout = pipelineLayoutHandle,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = 0};

    if (pvkCreateRayTracingPipelinesKHR(
            device.getDevice(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
            &rayTracingPipelineCreateInfo, NULL, &rayTracingPipelineHandle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create ray tracing pipeline!");
    }
}

void RayTracingPipeline::createShaderBindingTable()
{
    constexpr uint32_t ray_gen_count = 1;
    uint32_t miss_count = 2;
    uint32_t hit_count = 2;
    uint32_t handle_count = ray_gen_count + miss_count + hit_count;
    uint32_t handle_size = ray_tracing_properties.shaderGroupHandleSize;

    uint32_t handle_size_aligned = VulkanHelper::align_up(handle_size, ray_tracing_properties.shaderGroupHandleAlignment);

    rgenShaderBindingTable.stride = VulkanHelper::align_up(handle_size_aligned, ray_tracing_properties.shaderGroupBaseAlignment);
    rgenShaderBindingTable.size = rgenShaderBindingTable.stride;

    rmissShaderBindingTable.stride = handle_size_aligned;
    rmissShaderBindingTable.size = VulkanHelper::align_up(miss_count * handle_size_aligned, ray_tracing_properties.shaderGroupBaseAlignment);

    rchitShaderBindingTable.stride = handle_size_aligned;
    rchitShaderBindingTable.size = VulkanHelper::align_up(hit_count * handle_size_aligned, ray_tracing_properties.shaderGroupBaseAlignment);

    uint32_t dataSize = handle_count * handle_size;
    std::vector<uint8_t> handles(dataSize);
    auto result = pvkGetRayTracingShaderGroupHandlesKHR(device.getDevice(), rayTracingPipelineHandle, 0, handle_count, dataSize, handles.data());
    assert(result == VK_SUCCESS);

    VkDeviceSize sbtSize = rgenShaderBindingTable.size + rmissShaderBindingTable.size + rchitShaderBindingTable.size + callableShaderBindingTable.size;
    shader_binding_table = std::make_unique<Buffer>
    (
            device,
            sbtSize,
            1,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, shader_binding_table->getBuffer()};
    VkDeviceAddress sbtAddress = vkGetBufferDeviceAddress(device.getDevice(), &info);
    rgenShaderBindingTable.deviceAddress = sbtAddress;
    rmissShaderBindingTable.deviceAddress = sbtAddress + rgenShaderBindingTable.size;
    rchitShaderBindingTable.deviceAddress = sbtAddress + rgenShaderBindingTable.size + rmissShaderBindingTable.size;

    auto get_handle = [&] (uint32_t i) { return handles.data() + i * handle_size; };
    shader_binding_table->map();
    auto sbt_buffer = reinterpret_cast<uint8_t*>(shader_binding_table->getMappedMemory());
    uint8_t* data{nullptr};
    uint32_t handle_idx{0};

    data = sbt_buffer;
    memcpy(data, get_handle(handle_idx++), handle_size);

    data = sbt_buffer + rgenShaderBindingTable.size;
    for(uint32_t c = 0; c < hit_count; ++c)
    {
        memcpy(data, get_handle(handle_idx++), handle_size);
        data += rmissShaderBindingTable.stride;
    }

    data = sbt_buffer + rgenShaderBindingTable.size + rmissShaderBindingTable.size;
    for(uint32_t c = 0; c < miss_count; ++c)
    {
        memcpy(data, get_handle(handle_idx++), handle_size);
        data += rchitShaderBindingTable.stride;
    }
    shader_binding_table->unmap();
}

void RayTracingPipeline::bind(VkCommandBuffer command_buffer)
{
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rayTracingPipelineHandle);
}

void RayTracingPipeline::bindDescriptorSets(VkCommandBuffer command_buffer, const std::vector<VkDescriptorSet>& descriptor_sets)
{
    vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            pipelineLayoutHandle,
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
            pipelineLayoutHandle,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
            0,
            sizeof(PushConstantRay),
            &push_constant_ray);
}