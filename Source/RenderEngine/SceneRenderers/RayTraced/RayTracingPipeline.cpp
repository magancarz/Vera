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
    // Ray Closest Hit Shader Module

    std::ifstream rayClosestHitFile("Shaders/raytrace.rchit.spv",
                                    std::ios::binary | std::ios::ate);
    std::streamsize rayClosestHitFileSize = rayClosestHitFile.tellg();
    rayClosestHitFile.seekg(0, std::ios::beg);
    std::vector<uint32_t> rayClosestHitShaderSource(rayClosestHitFileSize /
                                                    sizeof(uint32_t));

    rayClosestHitFile.read(
            reinterpret_cast<char *>(rayClosestHitShaderSource.data()),
            rayClosestHitFileSize);

    rayClosestHitFile.close();

    VkShaderModuleCreateInfo rayClosestHitShaderModuleCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .codeSize = (uint32_t)rayClosestHitShaderSource.size() * sizeof(uint32_t),
            .pCode = rayClosestHitShaderSource.data()};

    VkShaderModule rayClosestHitShaderModuleHandle = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device.getDevice(), &rayClosestHitShaderModuleCreateInfo, NULL, &rayClosestHitShaderModuleHandle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module!");
    }

    ////////////////////////////
    std::ifstream rayClosestHitLightFile("Shaders/raytrace_light.rchit.spv",
                                    std::ios::binary | std::ios::ate);
    std::streamsize rayClosestHitLightFileSize = rayClosestHitLightFile.tellg();
    rayClosestHitLightFile.seekg(0, std::ios::beg);

    std::vector<uint32_t> rayClosestHitLightShaderSource(rayClosestHitLightFileSize / sizeof(uint32_t));
    rayClosestHitLightFile.read(reinterpret_cast<char *>(rayClosestHitLightShaderSource.data()), rayClosestHitLightFileSize);

    rayClosestHitLightFile.close();

    VkShaderModuleCreateInfo rayClosestHitLightShaderModuleCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .codeSize = (uint32_t)rayClosestHitLightShaderSource.size() * sizeof(uint32_t),
            .pCode = rayClosestHitLightShaderSource.data()};

    VkShaderModule rayClosestHitLightShaderModuleHandle = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device.getDevice(), &rayClosestHitLightShaderModuleCreateInfo, NULL, &rayClosestHitLightShaderModuleHandle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module!");
    }

    // =========================================================================
    // Ray Generate Shader Module

    std::ifstream rayGenerateFile("Shaders/raytrace.rgen.spv",
                                  std::ios::binary | std::ios::ate);
    std::streamsize rayGenerateFileSize = rayGenerateFile.tellg();
    rayGenerateFile.seekg(0, std::ios::beg);
    std::vector<uint32_t> rayGenerateShaderSource(rayGenerateFileSize /
                                                  sizeof(uint32_t));

    rayGenerateFile.read(reinterpret_cast<char *>(rayGenerateShaderSource.data()),
                         rayGenerateFileSize);

    rayGenerateFile.close();

    VkShaderModuleCreateInfo rayGenerateShaderModuleCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .codeSize = (uint32_t)rayGenerateShaderSource.size() * sizeof(uint32_t),
            .pCode = rayGenerateShaderSource.data()};

    VkShaderModule rayGenerateShaderModuleHandle = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device.getDevice(), &rayGenerateShaderModuleCreateInfo, NULL, &rayGenerateShaderModuleHandle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module!");
    }

    // =========================================================================
    // Ray Miss Shader Module

    std::ifstream rayMissFile("Shaders/raytrace.rmiss.spv",
                              std::ios::binary | std::ios::ate);
    std::streamsize rayMissFileSize = rayMissFile.tellg();
    rayMissFile.seekg(0, std::ios::beg);
    std::vector<uint32_t> rayMissShaderSource(rayMissFileSize / sizeof(uint32_t));

    rayMissFile.read(reinterpret_cast<char *>(rayMissShaderSource.data()),
                     rayMissFileSize);

    rayMissFile.close();

    VkShaderModuleCreateInfo rayMissShaderModuleCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .codeSize = (uint32_t)rayMissShaderSource.size() * sizeof(uint32_t),
            .pCode = rayMissShaderSource.data()};

    VkShaderModule rayMissShaderModuleHandle = VK_NULL_HANDLE;

    if (vkCreateShaderModule(device.getDevice(), &rayMissShaderModuleCreateInfo,
                             NULL, &rayMissShaderModuleHandle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module!");
    }

    // =========================================================================
    // Ray Miss Shader Module (Shadow)

    std::ifstream rayMissShadowFile("Shaders/raytrace_shadow.rmiss.spv",
                                    std::ios::binary | std::ios::ate);
    std::streamsize rayMissShadowFileSize = rayMissShadowFile.tellg();
    rayMissShadowFile.seekg(0, std::ios::beg);
    std::vector<uint32_t> rayMissShadowShaderSource(rayMissShadowFileSize / sizeof(uint32_t));
    rayMissShadowFile.read(
            reinterpret_cast<char *>(rayMissShadowShaderSource.data()),
            rayMissShadowFileSize);

    rayMissShadowFile.close();

    VkShaderModuleCreateInfo rayMissShadowShaderModuleCreateInfo =
    {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .codeSize = (uint32_t)rayMissShadowShaderSource.size() * sizeof(uint32_t),
            .pCode = rayMissShadowShaderSource.data()
    };

    VkShaderModule rayMissShadowShaderModuleHandle = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device.getDevice(), &rayMissShadowShaderModuleCreateInfo,
                             NULL, &rayMissShadowShaderModuleHandle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module!");
    }

    // =========================================================================
    // Ray Tracing Pipeline

    enum StageIndices
    {
        ray_gen,
        closest_hit_lambertian,
        closest_hit_light,
        miss,
        miss_shadow,
        shader_group_count
    };

    std::vector<VkPipelineShaderStageCreateInfo> pipeline_shader_stage_create_info_list;
    pipeline_shader_stage_create_info_list.reserve(shader_group_count);
    VkPipelineShaderStageCreateInfo stage_create_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_create_info.pName = "main";

    stage_create_info.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stage_create_info.module = rayGenerateShaderModuleHandle;
    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

    stage_create_info.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stage_create_info.module = rayClosestHitShaderModuleHandle;
    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

    stage_create_info.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stage_create_info.module = rayClosestHitLightShaderModuleHandle;
    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

    stage_create_info.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stage_create_info.module = rayMissShaderModuleHandle;
    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

    stage_create_info.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stage_create_info.module = rayMissShadowShaderModuleHandle;
    pipeline_shader_stage_create_info_list.push_back(stage_create_info);

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> ray_tracing_shader_group_create_info_list;

    VkRayTracingShaderGroupCreateInfoKHR ray_generate_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    ray_generate_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    ray_generate_shader_group_create_info.generalShader = ray_gen;
    ray_generate_shader_group_create_info.closestHitShader = VK_SHADER_UNUSED_KHR;
    ray_generate_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    ray_generate_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    ray_tracing_shader_group_create_info_list.push_back(ray_generate_shader_group_create_info);

    VkRayTracingShaderGroupCreateInfoKHR closest_hit_lambertian_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    closest_hit_lambertian_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    closest_hit_lambertian_shader_group_create_info.closestHitShader = closest_hit_lambertian;
    closest_hit_lambertian_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
    closest_hit_lambertian_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    closest_hit_lambertian_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    ray_tracing_shader_group_create_info_list.push_back(closest_hit_lambertian_shader_group_create_info);

    VkRayTracingShaderGroupCreateInfoKHR closest_hit_light_shader_group_create_info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    closest_hit_light_shader_group_create_info.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    closest_hit_light_shader_group_create_info.closestHitShader = closest_hit_light;
    closest_hit_light_shader_group_create_info.generalShader = VK_SHADER_UNUSED_KHR;
    closest_hit_light_shader_group_create_info.anyHitShader = VK_SHADER_UNUSED_KHR;
    closest_hit_light_shader_group_create_info.intersectionShader = VK_SHADER_UNUSED_KHR;
    ray_tracing_shader_group_create_info_list.push_back(closest_hit_light_shader_group_create_info);

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

//TODO: put it in some helper function
template <class integral>
constexpr integral align_up(integral x, size_t a) noexcept
{
    return integral((x + (integral(a) - 1)) & ~integral(a - 1));
}

void RayTracingPipeline::createShaderBindingTable()
{
    uint32_t miss_count = 2;
    uint32_t hit_count = 2;
    uint32_t handle_count = 1 + hit_count + miss_count;
    uint32_t handle_size  = ray_tracing_properties.shaderGroupHandleSize;

    uint32_t handle_size_aligned = align_up(handle_size, ray_tracing_properties.shaderGroupHandleAlignment);

    VkDeviceSize progSize = ray_tracing_properties.shaderGroupBaseAlignment;
    rgenShaderBindingTable.stride = align_up(handle_size_aligned, ray_tracing_properties.shaderGroupBaseAlignment);
    rgenShaderBindingTable.size = rgenShaderBindingTable.stride;

    rchitShaderBindingTable.stride = handle_size_aligned;
    rchitShaderBindingTable.size = align_up(hit_count * handle_size_aligned, ray_tracing_properties.shaderGroupBaseAlignment);

    rmissShaderBindingTable.stride = handle_size_aligned;
    rmissShaderBindingTable.size = align_up(miss_count * handle_size_aligned, ray_tracing_properties.shaderGroupBaseAlignment);

    uint32_t dataSize = handle_count * handle_size;
    std::vector<uint8_t> handles(dataSize);
    auto result = pvkGetRayTracingShaderGroupHandlesKHR(device.getDevice(), rayTracingPipelineHandle, 0, handle_count, dataSize, handles.data());
    assert(result == VK_SUCCESS);

    VkDeviceSize sbtSize = rgenShaderBindingTable.size + rchitShaderBindingTable.size + rmissShaderBindingTable.size + callableShaderBindingTable.size;
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
    rchitShaderBindingTable.deviceAddress = sbtAddress + rgenShaderBindingTable.size;
    rmissShaderBindingTable.deviceAddress = sbtAddress + rgenShaderBindingTable.size + rchitShaderBindingTable.size;

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
        data += rchitShaderBindingTable.stride;
    }

    data = sbt_buffer + rgenShaderBindingTable.size + rchitShaderBindingTable.size;
    for(uint32_t c = 0; c < miss_count; ++c)
    {
        memcpy(data, get_handle(handle_idx++), handle_size);
        data += rmissShaderBindingTable.stride;
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