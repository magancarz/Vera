#include "RayTracingAccelerationStructureBuilder.h"

#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

RayTracingAccelerationStructureBuilder::RayTracingAccelerationStructureBuilder(Device& device)
    : device{device} {}

AccelerationStructure RayTracingAccelerationStructureBuilder::buildBottomLevelAccelerationStructure(const BlasInput& blas_input)
{
    AccelerationStructure acceleration_structure{};

    VkAccelerationStructureBuildGeometryInfoKHR acceleration_structure_build_geometry_info{};
    acceleration_structure_build_geometry_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    acceleration_structure_build_geometry_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    acceleration_structure_build_geometry_info.flags = blas_input.flags;
    acceleration_structure_build_geometry_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    acceleration_structure_build_geometry_info.geometryCount = 1;
    acceleration_structure_build_geometry_info.pGeometries = &blas_input.acceleration_structure_geometry;

    std::vector<uint32_t> bottom_level_max_primitive_count_list = {blas_input.acceleration_structure_build_offset_info.primitiveCount};

    VkAccelerationStructureBuildSizesInfoKHR bottom_level_acceleration_structure_build_sizes_info{};
    bottom_level_acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    bottom_level_acceleration_structure_build_sizes_info.pNext = nullptr;
    bottom_level_acceleration_structure_build_sizes_info.accelerationStructureSize = 0;
    bottom_level_acceleration_structure_build_sizes_info.updateScratchSize = 0;
    bottom_level_acceleration_structure_build_sizes_info.buildScratchSize = 0;

    pvkGetAccelerationStructureBuildSizesKHR(
            device.getDevice(),
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &acceleration_structure_build_geometry_info,
            bottom_level_max_primitive_count_list.data(),
            &bottom_level_acceleration_structure_build_sizes_info);

    acceleration_structure.acceleration_structure_buffer = std::make_unique<Buffer>
    (
            device,
            bottom_level_acceleration_structure_build_sizes_info.accelerationStructureSize,
            1,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkAccelerationStructureCreateInfoKHR bottom_level_acceleration_structure_create_info{};
    bottom_level_acceleration_structure_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    bottom_level_acceleration_structure_create_info.createFlags = 0;
    bottom_level_acceleration_structure_create_info.buffer = acceleration_structure.acceleration_structure_buffer->getBuffer();
    bottom_level_acceleration_structure_create_info.offset = 0;
    bottom_level_acceleration_structure_create_info.size = bottom_level_acceleration_structure_build_sizes_info.accelerationStructureSize;
    bottom_level_acceleration_structure_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    VkAccelerationStructureKHR bottom_level_acceleration_structure_handle{VK_NULL_HANDLE};
    if (pvkCreateAccelerationStructureKHR(
            device.getDevice(),
            &bottom_level_acceleration_structure_create_info,
            VulkanDefines::NO_CALLBACK,
            &bottom_level_acceleration_structure_handle) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateAccelerationStructureKHR");
    }

    VkAccelerationStructureDeviceAddressInfoKHR bottom_level_acceleration_structure_device_address_info{};
    bottom_level_acceleration_structure_device_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    bottom_level_acceleration_structure_device_address_info.accelerationStructure = bottom_level_acceleration_structure_handle;

    acceleration_structure.bottom_level_acceleration_structure_device_address = pvkGetAccelerationStructureDeviceAddressKHR(device.getDevice(), &bottom_level_acceleration_structure_device_address_info);
    auto bottom_level_acceleration_structure_scratch_buffer = std::make_unique<Buffer>
    (
            device,
            bottom_level_acceleration_structure_build_sizes_info.accelerationStructureSize,
            1,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkDeviceAddress bottom_level_acceleration_structure_scratch_buffer_device_address = bottom_level_acceleration_structure_scratch_buffer->getBufferDeviceAddress();

    acceleration_structure_build_geometry_info.dstAccelerationStructure = bottom_level_acceleration_structure_handle;
    acceleration_structure_build_geometry_info.scratchData.deviceAddress = bottom_level_acceleration_structure_scratch_buffer_device_address;

    VkAccelerationStructureBuildRangeInfoKHR bottom_level_acceleration_structure_build_range_info{};
    bottom_level_acceleration_structure_build_range_info.primitiveCount = blas_input.acceleration_structure_build_offset_info.primitiveCount;
    bottom_level_acceleration_structure_build_range_info.primitiveOffset = 0;
    bottom_level_acceleration_structure_build_range_info.firstVertex = 0;
    bottom_level_acceleration_structure_build_range_info.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* bottom_level_acceleration_structure_build_range_infos = &bottom_level_acceleration_structure_build_range_info;

    VkCommandBuffer command_buffer = device.beginSingleTimeCommands();

    pvkCmdBuildAccelerationStructuresKHR(
            command_buffer, 1,
            &acceleration_structure_build_geometry_info,
            &bottom_level_acceleration_structure_build_range_infos);

    device.endSingleTimeCommands(command_buffer);

    acceleration_structure.acceleration_structure = acceleration_structure_build_geometry_info.dstAccelerationStructure;

    return acceleration_structure;
}

AccelerationStructure RayTracingAccelerationStructureBuilder::buildTopLevelAccelerationStructure(const std::vector<BlasInstance*>& blas_instances)
{
    AccelerationStructure tlas{};

    std::vector<VkAccelerationStructureInstanceKHR> instances_temp{};
    instances_temp.reserve(blas_instances.size());
    std::transform(blas_instances.begin(), blas_instances.end(), std::back_inserter(instances_temp),
                   [](const BlasInstance* blas_instance) { return blas_instance->bottomLevelAccelerationStructureInstance; });

    auto instances_buffer = std::make_unique<Buffer>
    (
        device,
        sizeof(VkAccelerationStructureInstanceKHR),
        static_cast<uint32_t>(blas_instances.size()),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    instances_buffer->writeWithStagingBuffer(instances_temp.data());

    VkAccelerationStructureGeometryInstancesDataKHR instances{};
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.arrayOfPointers = VK_FALSE;
    instances.data.deviceAddress = instances_buffer->getBufferDeviceAddress();

    VkAccelerationStructureGeometryDataKHR top_level_acceleration_structure_geometry_data{};
    top_level_acceleration_structure_geometry_data.instances = instances;

    VkAccelerationStructureGeometryKHR top_level_acceleration_structure_geometry{};
    top_level_acceleration_structure_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    top_level_acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    top_level_acceleration_structure_geometry.geometry = top_level_acceleration_structure_geometry_data;
    top_level_acceleration_structure_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR top_level_acceleration_structure_build_geometry_info{};
    top_level_acceleration_structure_build_geometry_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    top_level_acceleration_structure_build_geometry_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    top_level_acceleration_structure_build_geometry_info.flags = 0;
    top_level_acceleration_structure_build_geometry_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    top_level_acceleration_structure_build_geometry_info.srcAccelerationStructure = VK_NULL_HANDLE;
    top_level_acceleration_structure_build_geometry_info.dstAccelerationStructure = VK_NULL_HANDLE;
    top_level_acceleration_structure_build_geometry_info.geometryCount = 1;
    top_level_acceleration_structure_build_geometry_info.pGeometries = &top_level_acceleration_structure_geometry;

    VkAccelerationStructureBuildSizesInfoKHR top_level_acceleration_structure_build_sizes_info{};
    top_level_acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    top_level_acceleration_structure_build_sizes_info.accelerationStructureSize = 0;
    top_level_acceleration_structure_build_sizes_info.updateScratchSize = 0;
    top_level_acceleration_structure_build_sizes_info.buildScratchSize = 0;

    std::vector<uint32_t> top_level_max_primitive_count_list = {1};

    pvkGetAccelerationStructureBuildSizesKHR(
            device.getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &top_level_acceleration_structure_build_geometry_info,
            top_level_max_primitive_count_list.data(),
            &top_level_acceleration_structure_build_sizes_info);

    tlas.acceleration_structure_buffer = std::make_unique<Buffer>
    (
            device,
            top_level_acceleration_structure_build_sizes_info.accelerationStructureSize,
            1,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkAccelerationStructureCreateInfoKHR top_level_acceleration_structure_create_info{};
    top_level_acceleration_structure_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    top_level_acceleration_structure_create_info.createFlags = 0;
    top_level_acceleration_structure_create_info.buffer = tlas.acceleration_structure_buffer->getBuffer();
    top_level_acceleration_structure_create_info.offset = 0;
    top_level_acceleration_structure_create_info.size = top_level_acceleration_structure_build_sizes_info.accelerationStructureSize;
    top_level_acceleration_structure_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    top_level_acceleration_structure_create_info.deviceAddress = 0;


    if (pvkCreateAccelerationStructureKHR(
            device.getDevice(),
            &top_level_acceleration_structure_create_info,
            VulkanDefines::NO_CALLBACK,
            &tlas.acceleration_structure) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateAccelerationStructureKHR");
    }

    auto top_level_acceleration_structure_scratch_buffer = std::make_unique<Buffer>
    (
            device,
            top_level_acceleration_structure_build_sizes_info.buildScratchSize,
            1,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkDeviceAddress top_level_acceleration_structure_scratch_buffer_device_address = top_level_acceleration_structure_scratch_buffer->getBufferDeviceAddress();

    top_level_acceleration_structure_build_geometry_info.dstAccelerationStructure = tlas.acceleration_structure;
    top_level_acceleration_structure_build_geometry_info.scratchData.deviceAddress = top_level_acceleration_structure_scratch_buffer_device_address;

    VkAccelerationStructureBuildRangeInfoKHR top_level_acceleration_structure_build_range_info =
    {
            .primitiveCount = static_cast<uint32_t>(blas_instances.size()),
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0
    };

    const VkAccelerationStructureBuildRangeInfoKHR* top_level_acceleration_structure_build_range_infos = &top_level_acceleration_structure_build_range_info;

    VkCommandBuffer command_buffer = device.beginSingleTimeCommands();

    pvkCmdBuildAccelerationStructuresKHR(
            command_buffer, 1,
            &top_level_acceleration_structure_build_geometry_info,
            &top_level_acceleration_structure_build_range_infos);

    device.endSingleTimeCommands(command_buffer);

    return tlas;
}