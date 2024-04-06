#include "RayTracingBuilder.h"

#include "RenderEngine/RenderingAPI/VulkanDefines.h"

void throwExceptionVulkanAPI(VkResult result, const std::string &functionName) {
    std::string message = "Vulkan API exception: return code " +
                          std::to_string(result) + " (" + functionName + ")";

    throw std::runtime_error(message);
}

void RayTracingBuilder::setup(uint32_t in_queue_index, VkPhysicalDeviceRayTracingPipelinePropertiesKHR in_ray_tracing_properties)
{
    queue_index = in_queue_index;
    ray_tracing_properties = in_ray_tracing_properties;
}

void RayTracingBuilder::buildBlas(
        Device& device,
        const BlasInput& input,
        VkBuildAccelerationStructureFlagsKHR flags)
{
    VkResult result;

    VkAccelerationStructureBuildGeometryInfoKHR
            bottomLevelAccelerationStructureBuildGeometryInfo = {
            .sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .pNext = nullptr,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .flags = 0,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .srcAccelerationStructure = VK_NULL_HANDLE,
            .dstAccelerationStructure = VK_NULL_HANDLE,
            .geometryCount = 1,
            .pGeometries = &input.acceleration_structure_geometry,
            .ppGeometries = nullptr,
            .scratchData = {.deviceAddress = 0}};

    VkAccelerationStructureBuildSizesInfoKHR
            bottomLevelAccelerationStructureBuildSizesInfo = {
            .sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
            .pNext = nullptr,
            .accelerationStructureSize = 0,
            .updateScratchSize = 0,
            .buildScratchSize = 0};

    std::vector<uint32_t> bottomLevelMaxPrimitiveCountList = {input.acceleration_structure_build_offset_info.primitiveCount};

    vkGetAccelerationStructureBuildSizesKHR(
            device.getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &bottomLevelAccelerationStructureBuildGeometryInfo,
            bottomLevelMaxPrimitiveCountList.data(),
            &bottomLevelAccelerationStructureBuildSizesInfo);

    VkBufferCreateInfo bottom_level_acceleration_structure_buffer_create_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = bottomLevelAccelerationStructureBuildSizesInfo
                    .accelerationStructureSize,
            .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &queue_index};

    VkBuffer bottom_level_acceleration_structure_buffer_handle = VK_NULL_HANDLE;
    if (vkCreateBuffer(device.getDevice(), &bottom_level_acceleration_structure_buffer_create_info, VulkanDefines::NO_CALLBACK, &bottom_level_acceleration_structure_buffer_handle) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create buffer!");
    }

    VkMemoryRequirements bottomLevelAccelerationStructureMemoryRequirements;
    vkGetBufferMemoryRequirements(
            device.getDevice(), bottom_level_acceleration_structure_buffer_handle,
            &bottomLevelAccelerationStructureMemoryRequirements);

    uint32_t bottomLevelAccelerationStructureMemoryTypeIndex = -1;
    bottomLevelAccelerationStructureMemoryTypeIndex = device.findMemoryType(bottomLevelAccelerationStructureMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
//    for (uint32_t x = 0; x < physicalDeviceMemoryProperties.memoryTypeCount;
//         x++) {
//
//        if ((bottomLevelAccelerationStructureMemoryRequirements.memoryTypeBits &
//             (1 << x)) &&
//            (physicalDeviceMemoryProperties.memoryTypes[x].propertyFlags &
//             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ==
//            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
//
//            bottomLevelAccelerationStructureMemoryTypeIndex = x;
//            break;
//        }
//    }

    VkMemoryAllocateInfo bottomLevelAccelerationStructureMemoryAllocateInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = bottomLevelAccelerationStructureMemoryRequirements.size,
            .memoryTypeIndex = bottomLevelAccelerationStructureMemoryTypeIndex};

    VkDeviceMemory bottomLevelAccelerationStructureDeviceMemoryHandle =
            VK_NULL_HANDLE;

    result = vkAllocateMemory(
            device.getDevice(), &bottomLevelAccelerationStructureMemoryAllocateInfo, NULL,
            &bottomLevelAccelerationStructureDeviceMemoryHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkAllocateMemory");
    }

    result = vkBindBufferMemory(
            device.getDevice(), bottom_level_acceleration_structure_buffer_handle,
            bottomLevelAccelerationStructureDeviceMemoryHandle, 0);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkBindBufferMemory");
    }

    VkAccelerationStructureCreateInfoKHR
            bottomLevelAccelerationStructureCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .pNext = NULL,
            .createFlags = 0,
            .buffer = bottom_level_acceleration_structure_buffer_handle,
            .offset = 0,
            .size = bottomLevelAccelerationStructureBuildSizesInfo
                    .accelerationStructureSize,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .deviceAddress = 0};

    VkAccelerationStructureKHR bottomLevelAccelerationStructureHandle =
            VK_NULL_HANDLE;

    result = vkCreateAccelerationStructureKHR(
            device.getDevice(), &bottomLevelAccelerationStructureCreateInfo, NULL,
            &bottomLevelAccelerationStructureHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkCreateAccelerationStructureKHR");
    }

    // =========================================================================
    // Build Bottom Level Acceleration Structure

    VkAccelerationStructureDeviceAddressInfoKHR
            bottomLevelAccelerationStructureDeviceAddressInfo = {
            .sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .pNext = NULL,
            .accelerationStructure = bottomLevelAccelerationStructureHandle};

    VkDeviceAddress bottomLevelAccelerationStructureDeviceAddress =
            vkGetAccelerationStructureDeviceAddressKHR(
                    device.getDevice(), &bottomLevelAccelerationStructureDeviceAddressInfo);

    VkBufferCreateInfo bottomLevelAccelerationStructureScratchBufferCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .size = bottomLevelAccelerationStructureBuildSizesInfo.buildScratchSize,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &queue_index};

    VkBuffer bottomLevelAccelerationStructureScratchBufferHandle = VK_NULL_HANDLE;
    result = vkCreateBuffer(
            device.getDevice(), &bottomLevelAccelerationStructureScratchBufferCreateInfo,
            NULL, &bottomLevelAccelerationStructureScratchBufferHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkCreateBuffer");
    }

    VkMemoryRequirements
            bottomLevelAccelerationStructureScratchMemoryRequirements;
    vkGetBufferMemoryRequirements(
            device.getDevice(), bottomLevelAccelerationStructureScratchBufferHandle,
            &bottomLevelAccelerationStructureScratchMemoryRequirements);

    uint32_t bottomLevelAccelerationStructureScratchMemoryTypeIndex = device.findMemoryType(bottomLevelAccelerationStructureScratchMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
//    for (uint32_t x = 0; x < physicalDeviceMemoryProperties.memoryTypeCount;
//         x++) {
//
//        if ((bottomLevelAccelerationStructureScratchMemoryRequirements
//                     .memoryTypeBits &
//             (1 << x)) &&
//            (physicalDeviceMemoryProperties.memoryTypes[x].propertyFlags &
//             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ==
//            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
//
//            bottomLevelAccelerationStructureScratchMemoryTypeIndex = x;
//            break;
//        }
//    }

    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .pNext = NULL,
            .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            .deviceMask = 0};

    VkMemoryAllocateInfo
            bottomLevelAccelerationStructureScratchMemoryAllocateInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &memoryAllocateFlagsInfo,
            .allocationSize =
            bottomLevelAccelerationStructureScratchMemoryRequirements.size,
            .memoryTypeIndex =
            bottomLevelAccelerationStructureScratchMemoryTypeIndex};

    VkDeviceMemory bottomLevelAccelerationStructureDeviceScratchMemoryHandle =
            VK_NULL_HANDLE;

    result = vkAllocateMemory(
            device.getDevice(), &bottomLevelAccelerationStructureScratchMemoryAllocateInfo,
            NULL, &bottomLevelAccelerationStructureDeviceScratchMemoryHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkAllocateMemory");
    }

    result = vkBindBufferMemory(
            device.getDevice(), bottomLevelAccelerationStructureScratchBufferHandle,
            bottomLevelAccelerationStructureDeviceScratchMemoryHandle, 0);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkBindBufferMemory");
    }

    VkBufferDeviceAddressInfo
            bottomLevelAccelerationStructureScratchBufferDeviceAddressInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = NULL,
            .buffer = bottomLevelAccelerationStructureScratchBufferHandle};

    VkDeviceAddress bottomLevelAccelerationStructureScratchBufferDeviceAddress =
            vkGetBufferDeviceAddressKHR(
                    device.getDevice(),
                    &bottomLevelAccelerationStructureScratchBufferDeviceAddressInfo);

    bottomLevelAccelerationStructureBuildGeometryInfo.dstAccelerationStructure =
            bottomLevelAccelerationStructureHandle;

    bottomLevelAccelerationStructureBuildGeometryInfo.scratchData = {
            .deviceAddress =
            bottomLevelAccelerationStructureScratchBufferDeviceAddress};

    VkAccelerationStructureBuildRangeInfoKHR
            bottomLevelAccelerationStructureBuildRangeInfo = {.primitiveCount =
            input.acceleration_structure_build_offset_info.primitiveCount,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0};

    const VkAccelerationStructureBuildRangeInfoKHR
            *bottomLevelAccelerationStructureBuildRangeInfos =
            &bottomLevelAccelerationStructureBuildRangeInfo;

//    VkCommandBufferBeginInfo bottomLevelCommandBufferBeginInfo = {
//            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
//            .pNext = NULL,
//            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
//            .pInheritanceInfo = NULL};

    VkCommandBuffer command_buffer = device.beginSingleTimeCommands();
    vkCmdBuildAccelerationStructuresKHR(
            command_buffer, 1,
            &bottomLevelAccelerationStructureBuildGeometryInfo,
            &bottomLevelAccelerationStructureBuildRangeInfos);

    device.endSingleTimeCommands(command_buffer);

    VkSubmitInfo bottomLevelAccelerationStructureBuildSubmitInfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = NULL,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = NULL,
            .pWaitDstStageMask = NULL,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = NULL};

    VkFenceCreateInfo bottomLevelAccelerationStructureBuildFenceCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = NULL, .flags = 0};

    VkFence bottomLevelAccelerationStructureBuildFenceHandle = VK_NULL_HANDLE;
    result = vkCreateFence(
            device.getDevice(), &bottomLevelAccelerationStructureBuildFenceCreateInfo, NULL,
            &bottomLevelAccelerationStructureBuildFenceHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkCreateFence");
    }

    result = vkQueueSubmit(device.graphicsQueue(), 1,
                           &bottomLevelAccelerationStructureBuildSubmitInfo,
                           bottomLevelAccelerationStructureBuildFenceHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkQueueSubmit");
    }

    result = vkWaitForFences(device.getDevice(), 1,
                             &bottomLevelAccelerationStructureBuildFenceHandle,
                             true, UINT32_MAX);

    if (result != VK_SUCCESS && result != VK_TIMEOUT) {
        throwExceptionVulkanAPI(result, "vkWaitForFences");
    }
}