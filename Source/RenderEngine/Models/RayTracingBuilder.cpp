#include "RayTracingBuilder.h"

#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

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
        const BlasInput& input)
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

    pvkGetAccelerationStructureBuildSizesKHR(
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

    result = pvkCreateAccelerationStructureKHR(
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

    bottomLevelAccelerationStructureDeviceAddress =
            pvkGetAccelerationStructureDeviceAddressKHR(
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

    uint32_t bottomLevelAccelerationStructureScratchMemoryTypeIndex = device.findMemoryType(
            bottomLevelAccelerationStructureScratchMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

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
            vkGetBufferDeviceAddress(
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

    VkCommandBuffer command_buffer = device.beginSingleTimeCommands();

    pvkCmdBuildAccelerationStructuresKHR(
            command_buffer, 1,
            &bottomLevelAccelerationStructureBuildGeometryInfo,
            &bottomLevelAccelerationStructureBuildRangeInfos);

    device.endSingleTimeCommands(command_buffer);
}

void RayTracingBuilder::buildTlas(Device& device)
{
    VkResult result;

    VkAccelerationStructureInstanceKHR bottomLevelAccelerationStructureInstance =
            {.transform = {.matrix = {{1.0, 0.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0, 0.0},
                                      {0.0, 0.0, 1.0, 0.0}}},
                    .instanceCustomIndex = 0,
                    .mask = 0xFF,
                    .instanceShaderBindingTableRecordOffset = 0,
                    .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
                    .accelerationStructureReference =
                    bottomLevelAccelerationStructureDeviceAddress};

    VkBufferCreateInfo bottomLevelGeometryInstanceBufferCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .size = sizeof(VkAccelerationStructureInstanceKHR),
            .usage =
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &queue_index};

    VkBuffer bottomLevelGeometryInstanceBufferHandle = VK_NULL_HANDLE;
    result = vkCreateBuffer(device.getDevice(), &bottomLevelGeometryInstanceBufferCreateInfo,
                           NULL, &bottomLevelGeometryInstanceBufferHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkCreateBuffer");
    }

    VkMemoryRequirements bottomLevelGeometryInstanceMemoryRequirements;
    vkGetBufferMemoryRequirements(device.getDevice(),
                                  bottomLevelGeometryInstanceBufferHandle,
                                  &bottomLevelGeometryInstanceMemoryRequirements);

    uint32_t bottomLevelGeometryInstanceMemoryTypeIndex = device.findMemoryType(bottomLevelGeometryInstanceMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .pNext = NULL,
            .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            .deviceMask = 0};

    VkMemoryAllocateInfo bottomLevelGeometryInstanceMemoryAllocateInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &memoryAllocateFlagsInfo,
            .allocationSize = bottomLevelGeometryInstanceMemoryRequirements.size,
            .memoryTypeIndex = bottomLevelGeometryInstanceMemoryTypeIndex};

    VkDeviceMemory bottomLevelGeometryInstanceDeviceMemoryHandle = VK_NULL_HANDLE;

    result = vkAllocateMemory(
            device.getDevice(), &bottomLevelGeometryInstanceMemoryAllocateInfo, NULL,
            &bottomLevelGeometryInstanceDeviceMemoryHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkAllocateMemory");
    }

    result =
            vkBindBufferMemory(device.getDevice(), bottomLevelGeometryInstanceBufferHandle,
                               bottomLevelGeometryInstanceDeviceMemoryHandle, 0);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkBindBufferMemory");
    }

    void *hostbottomLevelGeometryInstanceMemoryBuffer;
    result =
            vkMapMemory(device.getDevice(), bottomLevelGeometryInstanceDeviceMemoryHandle,
                        0, sizeof(VkAccelerationStructureInstanceKHR), 0,
                        &hostbottomLevelGeometryInstanceMemoryBuffer);

    memcpy(hostbottomLevelGeometryInstanceMemoryBuffer,
           &bottomLevelAccelerationStructureInstance,
           sizeof(VkAccelerationStructureInstanceKHR));

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkMapMemory");
    }

    vkUnmapMemory(device.getDevice(), bottomLevelGeometryInstanceDeviceMemoryHandle);

    VkBufferDeviceAddressInfo bottomLevelGeometryInstanceDeviceAddressInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = NULL,
            .buffer = bottomLevelGeometryInstanceBufferHandle};

    VkDeviceAddress bottomLevelGeometryInstanceDeviceAddress =
            vkGetBufferDeviceAddress(
                    device.getDevice(), &bottomLevelGeometryInstanceDeviceAddressInfo);

    VkAccelerationStructureGeometryDataKHR topLevelAccelerationStructureGeometryData =
            {.instances = {
                    .sType =
                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                    .pNext = NULL,
                    .arrayOfPointers = VK_FALSE,
                    .data = {.deviceAddress =
                    bottomLevelGeometryInstanceDeviceAddress}}};

    VkAccelerationStructureGeometryKHR topLevelAccelerationStructureGeometry = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .pNext = NULL,
            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
            .geometry = topLevelAccelerationStructureGeometryData,
            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};

    VkAccelerationStructureBuildGeometryInfoKHR
            topLevelAccelerationStructureBuildGeometryInfo = {
            .sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .pNext = NULL,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            .flags = 0,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .srcAccelerationStructure = VK_NULL_HANDLE,
            .dstAccelerationStructure = VK_NULL_HANDLE,
            .geometryCount = 1,
            .pGeometries = &topLevelAccelerationStructureGeometry,
            .ppGeometries = NULL,
            .scratchData = {.deviceAddress = 0}};

    VkAccelerationStructureBuildSizesInfoKHR
            topLevelAccelerationStructureBuildSizesInfo = {
            .sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
            .pNext = NULL,
            .accelerationStructureSize = 0,
            .updateScratchSize = 0,
            .buildScratchSize = 0};

    std::vector<uint32_t> topLevelMaxPrimitiveCountList = {1};

    pvkGetAccelerationStructureBuildSizesKHR(
            device.getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &topLevelAccelerationStructureBuildGeometryInfo,
            topLevelMaxPrimitiveCountList.data(),
            &topLevelAccelerationStructureBuildSizesInfo);

    VkBufferCreateInfo topLevelAccelerationStructureBufferCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .size =
            topLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize,
            .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &queue_index};

    VkBuffer topLevelAccelerationStructureBufferHandle = VK_NULL_HANDLE;
    result = vkCreateBuffer(device.getDevice(),
                            &topLevelAccelerationStructureBufferCreateInfo, NULL,
                            &topLevelAccelerationStructureBufferHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkCreateBuffer");
    }

    VkMemoryRequirements topLevelAccelerationStructureMemoryRequirements;
    vkGetBufferMemoryRequirements(
            device.getDevice(), topLevelAccelerationStructureBufferHandle,
            &topLevelAccelerationStructureMemoryRequirements);

    uint32_t topLevelAccelerationStructureMemoryTypeIndex = device.findMemoryType(topLevelAccelerationStructureMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkMemoryAllocateInfo topLevelAccelerationStructureMemoryAllocateInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = NULL,
            .allocationSize = topLevelAccelerationStructureMemoryRequirements.size,
            .memoryTypeIndex = topLevelAccelerationStructureMemoryTypeIndex};

    VkDeviceMemory topLevelAccelerationStructureDeviceMemoryHandle =
            VK_NULL_HANDLE;

    result = vkAllocateMemory(
            device.getDevice(), &topLevelAccelerationStructureMemoryAllocateInfo, NULL,
            &topLevelAccelerationStructureDeviceMemoryHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkAllocateMemory");
    }

    result = vkBindBufferMemory(
            device.getDevice(), topLevelAccelerationStructureBufferHandle,
            topLevelAccelerationStructureDeviceMemoryHandle, 0);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkBindBufferMemory");
    }

    VkAccelerationStructureCreateInfoKHR topLevelAccelerationStructureCreateInfo =
            {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                    .pNext = NULL,
                    .createFlags = 0,
                    .buffer = topLevelAccelerationStructureBufferHandle,
                    .offset = 0,
                    .size = topLevelAccelerationStructureBuildSizesInfo
                            .accelerationStructureSize,
                    .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                    .deviceAddress = 0};

    result = pvkCreateAccelerationStructureKHR(
            device.getDevice(), &topLevelAccelerationStructureCreateInfo, NULL,
            &topLevelAccelerationStructureHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkCreateAccelerationStructureKHR");
    }

    // =========================================================================
    // Build Top Level Acceleration Structure

    VkAccelerationStructureDeviceAddressInfoKHR
            topLevelAccelerationStructureDeviceAddressInfo = {
            .sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .pNext = NULL,
            .accelerationStructure = topLevelAccelerationStructureHandle};

    VkDeviceAddress topLevelAccelerationStructureDeviceAddress =
            pvkGetAccelerationStructureDeviceAddressKHR(
                    device.getDevice(), &topLevelAccelerationStructureDeviceAddressInfo);

    VkBufferCreateInfo topLevelAccelerationStructureScratchBufferCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .size = topLevelAccelerationStructureBuildSizesInfo.buildScratchSize,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &queue_index};

    VkBuffer topLevelAccelerationStructureScratchBufferHandle = VK_NULL_HANDLE;
    result = vkCreateBuffer(
            device.getDevice(), &topLevelAccelerationStructureScratchBufferCreateInfo, NULL,
            &topLevelAccelerationStructureScratchBufferHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkCreateBuffer");
    }

    VkMemoryRequirements topLevelAccelerationStructureScratchMemoryRequirements;
    vkGetBufferMemoryRequirements(
            device.getDevice(), topLevelAccelerationStructureScratchBufferHandle,
            &topLevelAccelerationStructureScratchMemoryRequirements);

    uint32_t topLevelAccelerationStructureScratchMemoryTypeIndex = device.findMemoryType(topLevelAccelerationStructureScratchMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkMemoryAllocateInfo topLevelAccelerationStructureScratchMemoryAllocateInfo =
            {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .pNext = &memoryAllocateFlagsInfo,
                    .allocationSize =
                    topLevelAccelerationStructureScratchMemoryRequirements.size,
                    .memoryTypeIndex = topLevelAccelerationStructureScratchMemoryTypeIndex};

    VkDeviceMemory topLevelAccelerationStructureDeviceScratchMemoryHandle =
            VK_NULL_HANDLE;

    result = vkAllocateMemory(
            device.getDevice(), &topLevelAccelerationStructureScratchMemoryAllocateInfo,
            NULL, &topLevelAccelerationStructureDeviceScratchMemoryHandle);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkAllocateMemory");
    }

    result = vkBindBufferMemory(
            device.getDevice(), topLevelAccelerationStructureScratchBufferHandle,
            topLevelAccelerationStructureDeviceScratchMemoryHandle, 0);

    if (result != VK_SUCCESS) {
        throwExceptionVulkanAPI(result, "vkBindBufferMemory");
    }

    VkBufferDeviceAddressInfo
            topLevelAccelerationStructureScratchBufferDeviceAddressInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = NULL,
            .buffer = topLevelAccelerationStructureScratchBufferHandle};

    VkDeviceAddress topLevelAccelerationStructureScratchBufferDeviceAddress =
            vkGetBufferDeviceAddress(
                    device.getDevice(),
                    &topLevelAccelerationStructureScratchBufferDeviceAddressInfo);

    topLevelAccelerationStructureBuildGeometryInfo.dstAccelerationStructure =
            topLevelAccelerationStructureHandle;

    topLevelAccelerationStructureBuildGeometryInfo.scratchData = {
            .deviceAddress = topLevelAccelerationStructureScratchBufferDeviceAddress};

    VkAccelerationStructureBuildRangeInfoKHR
            topLevelAccelerationStructureBuildRangeInfo = {.primitiveCount = 1,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0};

    const VkAccelerationStructureBuildRangeInfoKHR
            *topLevelAccelerationStructureBuildRangeInfos =
            &topLevelAccelerationStructureBuildRangeInfo;

    VkCommandBuffer command_buffer = device.beginSingleTimeCommands();

    pvkCmdBuildAccelerationStructuresKHR(
            command_buffer, 1,
            &topLevelAccelerationStructureBuildGeometryInfo,
            &topLevelAccelerationStructureBuildRangeInfos);

    device.endSingleTimeCommands(command_buffer);
}