#include "RayTracingPipeline.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

#include <fstream>

RayTracingPipeline::RayTracingPipeline(Device& device, VkDescriptorSetLayout descriptor_set_layout, VkDescriptorSetLayout material_descriptor_set_layout,
       const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& ray_tracing_properties)
    : device{device}, descriptor_set_layout{descriptor_set_layout}, material_descriptor_set_layout{material_descriptor_set_layout},
      ray_tracing_properties{ray_tracing_properties}
{
    createPipeline();
}

void RayTracingPipeline::createPipeline()
{
    std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandleList = {descriptor_set_layout, material_descriptor_set_layout};

    VkResult result;
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .setLayoutCount = (uint32_t)descriptorSetLayoutHandleList.size(),
            .pSetLayouts = descriptorSetLayoutHandleList.data(),
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = NULL};

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

//    std::ifstream rayMissShadowFile("Shaders/shader.shadow.rmiss.spv",
//                                    std::ios::binary | std::ios::ate);
//    std::streamsize rayMissShadowFileSize = rayMissShadowFile.tellg();
//    rayMissShadowFile.seekg(0, std::ios::beg);
//    std::vector<uint32_t> rayMissShadowShaderSource(rayMissShadowFileSize /
//                                                    sizeof(uint32_t));
//
//    rayMissShadowFile.read(
//            reinterpret_cast<char *>(rayMissShadowShaderSource.data()),
//            rayMissShadowFileSize);
//
//    rayMissShadowFile.close();
//
//    VkShaderModuleCreateInfo rayMissShadowShaderModuleCreateInfo = {
//            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
//            .pNext = NULL,
//            .flags = 0,
//            .codeSize = (uint32_t)rayMissShadowShaderSource.size() * sizeof(uint32_t),
//            .pCode = rayMissShadowShaderSource.data()};
//
//    VkShaderModule rayMissShadowShaderModuleHandle = VK_NULL_HANDLE;
//    if (vkCreateShaderModule(device.getDevice(), &rayMissShadowShaderModuleCreateInfo,
//                             NULL, &rayMissShadowShaderModuleHandle) != VK_SUCCESS)
//    {
//        throw std::runtime_error("Failed to create shader module!");
//    }

    // =========================================================================
    // Ray Tracing Pipeline

    std::vector<VkPipelineShaderStageCreateInfo>
            pipelineShaderStageCreateInfoList = {
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = NULL,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                    .module = rayClosestHitShaderModuleHandle,
                    .pName = "main",
                    .pSpecializationInfo = NULL},
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = NULL,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                    .module = rayGenerateShaderModuleHandle,
                    .pName = "main",
                    .pSpecializationInfo = NULL},
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = NULL,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
                    .module = rayMissShaderModuleHandle,
                    .pName = "main",
                    .pSpecializationInfo = NULL}/*,
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = NULL,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
                    .module = rayMissShadowShaderModuleHandle,
                    .pName = "main",
                    .pSpecializationInfo = NULL}*/};

    std::vector<VkRayTracingShaderGroupCreateInfoKHR>
            rayTracingShaderGroupCreateInfoList = {
            {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                    .pNext = NULL,
                    .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                    .generalShader = VK_SHADER_UNUSED_KHR,
                    .closestHitShader = 0,
                    .anyHitShader = VK_SHADER_UNUSED_KHR,
                    .intersectionShader = VK_SHADER_UNUSED_KHR,
                    .pShaderGroupCaptureReplayHandle = NULL},
            {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                    .pNext = NULL,
                    .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                    .generalShader = 1,
                    .closestHitShader = VK_SHADER_UNUSED_KHR,
                    .anyHitShader = VK_SHADER_UNUSED_KHR,
                    .intersectionShader = VK_SHADER_UNUSED_KHR,
                    .pShaderGroupCaptureReplayHandle = NULL},
            {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                    .pNext = NULL,
                    .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                    .generalShader = 2,
                    .closestHitShader = VK_SHADER_UNUSED_KHR,
                    .anyHitShader = VK_SHADER_UNUSED_KHR,
                    .intersectionShader = VK_SHADER_UNUSED_KHR,
                    .pShaderGroupCaptureReplayHandle = NULL}/*,
            {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                    .pNext = NULL,
                    .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                    .generalShader = 3,
                    .closestHitShader = VK_SHADER_UNUSED_KHR,
                    .anyHitShader = VK_SHADER_UNUSED_KHR,
                    .intersectionShader = VK_SHADER_UNUSED_KHR,
                    .pShaderGroupCaptureReplayHandle = NULL}*/};

    VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            .pNext = NULL,
            .flags = 0,
            .stageCount = static_cast<uint32_t>(pipelineShaderStageCreateInfoList.size()),
            .pStages = pipelineShaderStageCreateInfoList.data(),
            .groupCount = static_cast<uint32_t>(rayTracingShaderGroupCreateInfoList.size()),
            .pGroups = rayTracingShaderGroupCreateInfoList.data(),
            .maxPipelineRayRecursionDepth = 1,
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

    VkDeviceSize progSize = ray_tracing_properties.shaderGroupBaseAlignment;

    VkDeviceSize shaderBindingTableSize = progSize * 4;

    auto queue_index = device.findPhysicalQueueFamilies().graphicsFamily;
    VkBufferCreateInfo shaderBindingTableBufferCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .size = shaderBindingTableSize,
            .usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &queue_index};

    VkBuffer shaderBindingTableBufferHandle = VK_NULL_HANDLE;

    if (vkCreateBuffer(device.getDevice(), &shaderBindingTableBufferCreateInfo,
                       NULL, &shaderBindingTableBufferHandle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create buffer");
    }

    VkMemoryRequirements shaderBindingTableMemoryRequirements;
    vkGetBufferMemoryRequirements(device.getDevice(), shaderBindingTableBufferHandle,
                                  &shaderBindingTableMemoryRequirements);

    uint32_t shaderBindingTableMemoryTypeIndex = device.findMemoryType(shaderBindingTableMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .pNext = NULL,
            .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            .deviceMask = 0};

    VkMemoryAllocateInfo shaderBindingTableMemoryAllocateInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &memoryAllocateFlagsInfo,
            .allocationSize = shaderBindingTableMemoryRequirements.size,
            .memoryTypeIndex = shaderBindingTableMemoryTypeIndex};

    VkDeviceMemory shaderBindingTableDeviceMemoryHandle = VK_NULL_HANDLE;
    if (vkAllocateMemory(device.getDevice(), &shaderBindingTableMemoryAllocateInfo,
                         NULL, &shaderBindingTableDeviceMemoryHandle) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate memory");
    }

    if (vkBindBufferMemory(device.getDevice(), shaderBindingTableBufferHandle,
                           shaderBindingTableDeviceMemoryHandle, 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to bind buffer memory");
    }

    char *shaderHandleBuffer = new char[shaderBindingTableSize];
    if (pvkGetRayTracingShaderGroupHandlesKHR(
            device.getDevice(), rayTracingPipelineHandle, 0, 3, shaderBindingTableSize,
            shaderHandleBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to get ray tracing shader group handles");
    }

    void *hostShaderBindingTableMemoryBuffer;
    result = vkMapMemory(device.getDevice(), shaderBindingTableDeviceMemoryHandle, 0,
                         shaderBindingTableSize, 0,
                         &hostShaderBindingTableMemoryBuffer);

    for (uint32_t x = 0; x < 4; x++) {
        memcpy(hostShaderBindingTableMemoryBuffer,
               shaderHandleBuffer + x * ray_tracing_properties
                       .shaderGroupHandleSize,
               ray_tracing_properties.shaderGroupHandleSize);

        hostShaderBindingTableMemoryBuffer =
                reinterpret_cast<char *>(hostShaderBindingTableMemoryBuffer) +
                        ray_tracing_properties.shaderGroupBaseAlignment;
    }

    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to map memory");
    }

    vkUnmapMemory(device.getDevice(), shaderBindingTableDeviceMemoryHandle);

    VkBufferDeviceAddressInfo shaderBindingTableBufferDeviceAddressInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = NULL,
            .buffer = shaderBindingTableBufferHandle};

    shaderBindingTableBufferDeviceAddress = vkGetBufferDeviceAddress(
            device.getDevice(),
            &shaderBindingTableBufferDeviceAddressInfo);

    hitGroupOffset = 0u * progSize;
    rayGenOffset = 1u * progSize;
    missOffset = 2u * progSize;

    rchitShaderBindingTable = VkStridedDeviceAddressRegionKHR{
            .deviceAddress = shaderBindingTableBufferDeviceAddress + hitGroupOffset,
            .stride = progSize,
            .size = progSize};

    rgenShaderBindingTable = VkStridedDeviceAddressRegionKHR{
            .deviceAddress = shaderBindingTableBufferDeviceAddress + rayGenOffset,
            .stride = progSize,
            .size = progSize};

    rmissShaderBindingTable = VkStridedDeviceAddressRegionKHR{
            .deviceAddress = shaderBindingTableBufferDeviceAddress + missOffset,
            .stride = progSize,
            .size = progSize * 2};
}

void RayTracingPipeline::bind(VkCommandBuffer command_buffer)
{
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rayTracingPipelineHandle);
}

void RayTracingPipeline::bindDescriptorSets(VkCommandBuffer command_buffer, const std::vector<VkDescriptorSet>& descriptor_sets)
{
    vkCmdBindDescriptorSets(
            command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            pipelineLayoutHandle, 0, (uint32_t)descriptor_sets.size(),
            descriptor_sets.data(), 0, nullptr);
}