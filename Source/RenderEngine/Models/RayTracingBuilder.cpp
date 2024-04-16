#include "RayTracingAccelerationStructureBuilder.h"

#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

RayTracingAccelerationStructureBuilder::RayTracingAccelerationStructureBuilder(Device& device)
    : device{device} {}

AccelerationStructure RayTracingAccelerationStructureBuilder::buildBottomLevelAccelerationStructure(const BlasInput& blas_input)
{
    AccelerationStructure acceleration_structure{};

    VkAccelerationStructureBuildGeometryInfoKHR bottomLevelAccelerationStructureBuildGeometryInfo{};
    bottomLevelAccelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    bottomLevelAccelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bottomLevelAccelerationStructureBuildGeometryInfo.flags = blas_input.flags;
    bottomLevelAccelerationStructureBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bottomLevelAccelerationStructureBuildGeometryInfo.geometryCount = 1;
    bottomLevelAccelerationStructureBuildGeometryInfo.pGeometries = &blas_input.acceleration_structure_geometry;

    std::vector<uint32_t> bottomLevelMaxPrimitiveCountList = {blas_input.acceleration_structure_build_offset_info.primitiveCount};

    VkAccelerationStructureBuildSizesInfoKHR bottomLevelAccelerationStructureBuildSizesInfo{};
    bottomLevelAccelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    bottomLevelAccelerationStructureBuildSizesInfo.pNext = nullptr;
    bottomLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize = 0;
    bottomLevelAccelerationStructureBuildSizesInfo.updateScratchSize = 0;
    bottomLevelAccelerationStructureBuildSizesInfo.buildScratchSize = 0;

    pvkGetAccelerationStructureBuildSizesKHR(
            device.getDevice(),
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &bottomLevelAccelerationStructureBuildGeometryInfo,
            bottomLevelMaxPrimitiveCountList.data(),
            &bottomLevelAccelerationStructureBuildSizesInfo);

    acceleration_structure.acceleration_structure_buffer = std::make_unique<Buffer>
    (
            device,
            bottomLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize,
            1,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkAccelerationStructureCreateInfoKHR bottomLevelAccelerationStructureCreateInfo{};
    bottomLevelAccelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    bottomLevelAccelerationStructureCreateInfo.createFlags = 0;
    bottomLevelAccelerationStructureCreateInfo.buffer = acceleration_structure.acceleration_structure_buffer->getBuffer();
    bottomLevelAccelerationStructureCreateInfo.offset = 0;
    bottomLevelAccelerationStructureCreateInfo.size = bottomLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize;
    bottomLevelAccelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    VkAccelerationStructureKHR bottomLevelAccelerationStructureHandle = VK_NULL_HANDLE;
    if (pvkCreateAccelerationStructureKHR(
            device.getDevice(),
            &bottomLevelAccelerationStructureCreateInfo,
            VulkanDefines::NO_CALLBACK,
            &bottomLevelAccelerationStructureHandle) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateAccelerationStructureKHR");
    }

    VkAccelerationStructureDeviceAddressInfoKHR bottomLevelAccelerationStructureDeviceAddressInfo{};
    bottomLevelAccelerationStructureDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    bottomLevelAccelerationStructureDeviceAddressInfo.accelerationStructure = bottomLevelAccelerationStructureHandle;

    acceleration_structure.bottom_level_acceleration_structure_device_address = pvkGetAccelerationStructureDeviceAddressKHR(device.getDevice(), &bottomLevelAccelerationStructureDeviceAddressInfo);
    auto bottom_level_acceleration_structure_scratch_buffer = std::make_unique<Buffer>
    (
            device,
            bottomLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize,
            1,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkDeviceAddress bottomLevelAccelerationStructureScratchBufferDeviceAddress = bottom_level_acceleration_structure_scratch_buffer->getBufferDeviceAddress();

    bottomLevelAccelerationStructureBuildGeometryInfo.dstAccelerationStructure = bottomLevelAccelerationStructureHandle;
    bottomLevelAccelerationStructureBuildGeometryInfo.scratchData.deviceAddress = bottomLevelAccelerationStructureScratchBufferDeviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR bottomLevelAccelerationStructureBuildRangeInfo{};
    bottomLevelAccelerationStructureBuildRangeInfo.primitiveCount = blas_input.acceleration_structure_build_offset_info.primitiveCount;
    bottomLevelAccelerationStructureBuildRangeInfo.primitiveOffset = 0;
    bottomLevelAccelerationStructureBuildRangeInfo.firstVertex = 0;
    bottomLevelAccelerationStructureBuildRangeInfo.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* bottomLevelAccelerationStructureBuildRangeInfos = &bottomLevelAccelerationStructureBuildRangeInfo;

    VkCommandBuffer command_buffer = device.beginSingleTimeCommands();

    pvkCmdBuildAccelerationStructuresKHR(
            command_buffer, 1,
            &bottomLevelAccelerationStructureBuildGeometryInfo,
            &bottomLevelAccelerationStructureBuildRangeInfos);

    device.endSingleTimeCommands(command_buffer);

    return acceleration_structure;
}

AccelerationStructure RayTracingAccelerationStructureBuilder::buildTopLevelAccelerationStructure(const AccelerationStructure& blas_structure)
{
    AccelerationStructure tlas{};

    VkTransformMatrixKHR transform = VulkanHelper::mat4ToVkTransformMatrixKHR(glm::mat4{1.f});

    VkAccelerationStructureInstanceKHR bottomLevelAccelerationStructureInstance{};
    bottomLevelAccelerationStructureInstance.transform = transform;
    bottomLevelAccelerationStructureInstance.instanceCustomIndex = 0;
    bottomLevelAccelerationStructureInstance.mask = 0xFF;
    bottomLevelAccelerationStructureInstance.instanceShaderBindingTableRecordOffset = 0;
    bottomLevelAccelerationStructureInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    bottomLevelAccelerationStructureInstance.accelerationStructureReference = blas_structure.bottom_level_acceleration_structure_device_address;

    auto bottom_level_geometry_instance_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(VkAccelerationStructureInstanceKHR),
            1,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    );
    bottom_level_geometry_instance_buffer->map();
    bottom_level_geometry_instance_buffer->writeToBuffer(&bottomLevelAccelerationStructureInstance);
    bottom_level_geometry_instance_buffer->unmap();

    VkDeviceAddress bottomLevelGeometryInstanceDeviceAddress = bottom_level_geometry_instance_buffer->getBufferDeviceAddress();

    VkAccelerationStructureGeometryInstancesDataKHR instances{};
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.arrayOfPointers = VK_FALSE;
    instances.data.deviceAddress = bottomLevelGeometryInstanceDeviceAddress;

    VkAccelerationStructureGeometryDataKHR topLevelAccelerationStructureGeometryData{};
    topLevelAccelerationStructureGeometryData.instances = instances;

    VkAccelerationStructureGeometryKHR topLevelAccelerationStructureGeometry{};
    topLevelAccelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    topLevelAccelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    topLevelAccelerationStructureGeometry.geometry = topLevelAccelerationStructureGeometryData;
    topLevelAccelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    std::vector<VkAccelerationStructureGeometryKHR> topLevelAccelerationStructureGeometries = {topLevelAccelerationStructureGeometry};

    VkAccelerationStructureBuildGeometryInfoKHR topLevelAccelerationStructureBuildGeometryInfo{};
    topLevelAccelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    topLevelAccelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    topLevelAccelerationStructureBuildGeometryInfo.flags = 0;
    topLevelAccelerationStructureBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    topLevelAccelerationStructureBuildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    topLevelAccelerationStructureBuildGeometryInfo.dstAccelerationStructure = VK_NULL_HANDLE;
    topLevelAccelerationStructureBuildGeometryInfo.geometryCount = static_cast<uint32_t>(topLevelAccelerationStructureGeometries.size());
    topLevelAccelerationStructureBuildGeometryInfo.pGeometries = topLevelAccelerationStructureGeometries.data();

    VkAccelerationStructureBuildSizesInfoKHR topLevelAccelerationStructureBuildSizesInfo{};
    topLevelAccelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    topLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize = 0;
    topLevelAccelerationStructureBuildSizesInfo.updateScratchSize = 0;
    topLevelAccelerationStructureBuildSizesInfo.buildScratchSize = 0;

    std::vector<uint32_t> topLevelMaxPrimitiveCountList = {1};

    pvkGetAccelerationStructureBuildSizesKHR(
            device.getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &topLevelAccelerationStructureBuildGeometryInfo,
            topLevelMaxPrimitiveCountList.data(),
            &topLevelAccelerationStructureBuildSizesInfo);

    tlas.acceleration_structure_buffer = std::make_unique<Buffer>
    (
            device,
            topLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize,
            1,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkAccelerationStructureCreateInfoKHR topLevelAccelerationStructureCreateInfo{};
    topLevelAccelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    topLevelAccelerationStructureCreateInfo.createFlags = 0;
    topLevelAccelerationStructureCreateInfo.buffer = tlas.acceleration_structure_buffer->getBuffer();
    topLevelAccelerationStructureCreateInfo.offset = 0;
    topLevelAccelerationStructureCreateInfo.size = topLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize;
    topLevelAccelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    topLevelAccelerationStructureCreateInfo.deviceAddress = 0;


    if (pvkCreateAccelerationStructureKHR(
            device.getDevice(),
            &topLevelAccelerationStructureCreateInfo,
            VulkanDefines::NO_CALLBACK,
            &tlas.acceleration_structure) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateAccelerationStructureKHR");
    }

    auto top_level_acceleration_structure_scratch_buffer = std::make_unique<Buffer>
    (
            device,
            topLevelAccelerationStructureBuildSizesInfo.buildScratchSize,
            1,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkDeviceAddress topLevelAccelerationStructureScratchBufferDeviceAddress = top_level_acceleration_structure_scratch_buffer->getBufferDeviceAddress();

    topLevelAccelerationStructureBuildGeometryInfo.dstAccelerationStructure = tlas.acceleration_structure;
    topLevelAccelerationStructureBuildGeometryInfo.scratchData.deviceAddress = topLevelAccelerationStructureScratchBufferDeviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR topLevelAccelerationStructureBuildRangeInfo =
    {
            .primitiveCount = 1,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0
    };

    const VkAccelerationStructureBuildRangeInfoKHR* topLevelAccelerationStructureBuildRangeInfos = &topLevelAccelerationStructureBuildRangeInfo;

    VkCommandBuffer command_buffer = device.beginSingleTimeCommands();

    pvkCmdBuildAccelerationStructuresKHR(
            command_buffer, 1,
            &topLevelAccelerationStructureBuildGeometryInfo,
            &topLevelAccelerationStructureBuildRangeInfos);

    device.endSingleTimeCommands(command_buffer);

    return tlas;
}