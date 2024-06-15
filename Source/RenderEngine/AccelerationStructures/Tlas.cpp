#include "Tlas.h"

#include <RenderEngine/RenderingAPI/VulkanDefines.h>

#include "Memory/MemoryAllocator.h"
#include "BlasInstance.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

Tlas::Tlas(Device& logical_device, CommandPool& command_pool, MemoryAllocator& memory_allocator, VkBuildAccelerationStructureFlagsKHR build_flags)
    : logical_device{logical_device}, command_pool{command_pool}, memory_allocator{memory_allocator}, build_flags{build_flags} {}

void Tlas::build(const std::vector<BlasInstance>& blas_instances)
{
    copyBlasInstancesDataToBuffer(blas_instances);

    //TODO: find out why it doesnt work when it is in a function
    // auto [geometries, build_geometry_info] = fillBuildGeometryInfo(VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR);
    VkAccelerationStructureGeometryInstancesDataKHR instances{};
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.arrayOfPointers = VK_FALSE;
    instances.data.deviceAddress = blas_instances_buffer->getBufferDeviceAddress();

    VkAccelerationStructureGeometryDataKHR acceleration_structure_geometry_data{};
    acceleration_structure_geometry_data.instances = instances;

    VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
    acceleration_structure_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    acceleration_structure_geometry.geometry = acceleration_structure_geometry_data;
    acceleration_structure_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR acceleration_structure_build_geometry_info{};
    acceleration_structure_build_geometry_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    acceleration_structure_build_geometry_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    acceleration_structure_build_geometry_info.flags = build_flags;
    acceleration_structure_build_geometry_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    acceleration_structure_build_geometry_info.srcAccelerationStructure = VK_NULL_HANDLE;
    acceleration_structure_build_geometry_info.dstAccelerationStructure = VK_NULL_HANDLE;
    acceleration_structure_build_geometry_info.geometryCount = 1;
    acceleration_structure_build_geometry_info.pGeometries = &acceleration_structure_geometry;

    VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info = obtainBuildSizesInfo(acceleration_structure_build_geometry_info, blas_instances.size());
    createAccelerationStructure(acceleration_structure_build_sizes_info.accelerationStructureSize);
    std::unique_ptr<Buffer> scratch_buffer = createScratchBuffer(acceleration_structure_build_sizes_info.buildScratchSize);

    acceleration_structure_build_geometry_info.dstAccelerationStructure = acceleration_structure.handle;
    acceleration_structure_build_geometry_info.scratchData.deviceAddress = scratch_buffer->getBufferDeviceAddress();

    VkAccelerationStructureBuildRangeInfoKHR top_level_acceleration_structure_build_range_info =
    {
        .primitiveCount = static_cast<uint32_t>(blas_instances.size()),
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0
    };

    cmdBuildTlas(acceleration_structure_build_geometry_info, &top_level_acceleration_structure_build_range_info);
}

void Tlas::copyBlasInstancesDataToBuffer(const std::vector<BlasInstance>& blas_instances)
{
    if (!blas_instances_buffer)
    {
        createBlasInstancesBuffer(blas_instances.size());
    }

    if (!blas_instances_staging_buffer)
    {
        blas_instances_staging_buffer = memory_allocator.createStagingBuffer(
        sizeof(VkAccelerationStructureInstanceKHR), blas_instances.size());
    }

    std::vector<VkAccelerationStructureInstanceKHR> instances_temp{};
    instances_temp.reserve(blas_instances.size());
    std::ranges::transform(blas_instances.begin(), blas_instances.end(), std::back_inserter(instances_temp),
        [](const BlasInstance& blas_instance)
        {
           return blas_instance.bottom_level_acceleration_structure_instance;
        });

    blas_instances_staging_buffer->writeToBuffer(instances_temp.data());
    blas_instances_buffer->copyFrom(*blas_instances_staging_buffer);
}

void Tlas::createBlasInstancesBuffer(uint32_t blas_instances_count)
{
    BufferInfo instances_buffer_info{};
    instances_buffer_info.instance_size = sizeof(VkAccelerationStructureInstanceKHR);
    instances_buffer_info.instance_count = blas_instances_count;
    instances_buffer_info.usage_flags =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    instances_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    blas_instances_buffer = memory_allocator.createBuffer(instances_buffer_info);
}

std::pair<VkAccelerationStructureGeometryKHR, VkAccelerationStructureBuildGeometryInfoKHR> Tlas::fillBuildGeometryInfo(VkBuildAccelerationStructureModeKHR mode)
{
    VkAccelerationStructureGeometryInstancesDataKHR instances{};
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.arrayOfPointers = VK_FALSE;
    instances.data.deviceAddress = blas_instances_buffer->getBufferDeviceAddress();

    VkAccelerationStructureGeometryDataKHR acceleration_structure_geometry_data{};
    acceleration_structure_geometry_data.instances = instances;

    VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
    acceleration_structure_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    acceleration_structure_geometry.geometry = acceleration_structure_geometry_data;
    acceleration_structure_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR acceleration_structure_build_geometry_info{};
    acceleration_structure_build_geometry_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    acceleration_structure_build_geometry_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    acceleration_structure_build_geometry_info.flags = build_flags;
    acceleration_structure_build_geometry_info.mode = mode;
    acceleration_structure_build_geometry_info.srcAccelerationStructure = VK_NULL_HANDLE;
    acceleration_structure_build_geometry_info.dstAccelerationStructure = VK_NULL_HANDLE;
    acceleration_structure_build_geometry_info.geometryCount = 1;
    acceleration_structure_build_geometry_info.pGeometries = &acceleration_structure_geometry;

    return {acceleration_structure_geometry, acceleration_structure_build_geometry_info};
}

VkAccelerationStructureBuildSizesInfoKHR Tlas::obtainBuildSizesInfo(const VkAccelerationStructureBuildGeometryInfoKHR& build_geometry_info, uint32_t blas_instances_count)
{
    VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{};
    acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    acceleration_structure_build_sizes_info.accelerationStructureSize = 0;
    acceleration_structure_build_sizes_info.updateScratchSize = 0;
    acceleration_structure_build_sizes_info.buildScratchSize = 0;

    std::vector<uint32_t> max_primitive_count_list{blas_instances_count};

    pvkGetAccelerationStructureBuildSizesKHR(
        logical_device.getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_geometry_info,
        max_primitive_count_list.data(),
        &acceleration_structure_build_sizes_info);

    return acceleration_structure_build_sizes_info;
}

void Tlas::createAccelerationStructure(uint32_t acceleration_structure_size)
{
    BufferInfo acceleration_structure_buffer_info{};
    acceleration_structure_buffer_info.instance_size = acceleration_structure_size;
    acceleration_structure_buffer_info.instance_count = 1;
    acceleration_structure_buffer_info.usage_flags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    acceleration_structure_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    acceleration_structure.buffer = memory_allocator.createBuffer(acceleration_structure_buffer_info);

    VkAccelerationStructureCreateInfoKHR top_level_acceleration_structure_create_info{};
    top_level_acceleration_structure_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    top_level_acceleration_structure_create_info.createFlags = 0;
    top_level_acceleration_structure_create_info.buffer = acceleration_structure.buffer->getBuffer();
    top_level_acceleration_structure_create_info.offset = 0;
    top_level_acceleration_structure_create_info.size = acceleration_structure_size;
    top_level_acceleration_structure_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    top_level_acceleration_structure_create_info.deviceAddress = 0;

    if (pvkCreateAccelerationStructureKHR(
        logical_device.getDevice(),
        &top_level_acceleration_structure_create_info,
        VulkanDefines::NO_CALLBACK,
        &acceleration_structure.handle) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateAccelerationStructureKHR");
    }
}

std::unique_ptr<Buffer> Tlas::createScratchBuffer(uint32_t scratch_buffer_size)
{
    BufferInfo acceleration_structure_scratch_buffer_info{};
    acceleration_structure_scratch_buffer_info.instance_size = scratch_buffer_size;
    acceleration_structure_scratch_buffer_info.instance_count = 1;
    acceleration_structure_scratch_buffer_info.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    acceleration_structure_scratch_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    return memory_allocator.createBuffer(acceleration_structure_scratch_buffer_info);
}

void Tlas::cmdBuildTlas(const VkAccelerationStructureBuildGeometryInfoKHR& build_geometry_info, VkAccelerationStructureBuildRangeInfoKHR* build_range_info)
{
    VkCommandBuffer command_buffer = command_pool.beginSingleTimeCommands();

    pvkCmdBuildAccelerationStructuresKHR(
        command_buffer, 1, &build_geometry_info, &build_range_info);

    command_pool.endSingleTimeCommands(command_buffer);
}

void Tlas::update(const std::vector<BlasInstance>& blas_instances)
{
    assert(build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR &&
        "Cannot call update if tlas wasn't created with VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR flag");

    copyBlasInstancesDataToBuffer(blas_instances);
    // auto [geometries, build_geometry_info] = fillBuildGeometryInfo(VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR);

    //TODO: find out why it doesnt work when it is in a function
    VkAccelerationStructureGeometryInstancesDataKHR instances{};
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.arrayOfPointers = VK_FALSE;
    instances.data.deviceAddress = blas_instances_buffer->getBufferDeviceAddress();

    VkAccelerationStructureGeometryDataKHR acceleration_structure_geometry_data{};
    acceleration_structure_geometry_data.instances = instances;

    VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
    acceleration_structure_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    acceleration_structure_geometry.geometry = acceleration_structure_geometry_data;
    acceleration_structure_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR acceleration_structure_build_geometry_info{};
    acceleration_structure_build_geometry_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    acceleration_structure_build_geometry_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    acceleration_structure_build_geometry_info.flags = build_flags;
    acceleration_structure_build_geometry_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    acceleration_structure_build_geometry_info.srcAccelerationStructure = VK_NULL_HANDLE;
    acceleration_structure_build_geometry_info.dstAccelerationStructure = VK_NULL_HANDLE;
    acceleration_structure_build_geometry_info.geometryCount = 1;
    acceleration_structure_build_geometry_info.pGeometries = &acceleration_structure_geometry;

    VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info = obtainBuildSizesInfo(acceleration_structure_build_geometry_info, blas_instances.size());
    std::unique_ptr<Buffer> scratch_buffer = createScratchBuffer(acceleration_structure_build_sizes_info.buildScratchSize);

    acceleration_structure_build_geometry_info.srcAccelerationStructure = acceleration_structure.handle;
    acceleration_structure_build_geometry_info.dstAccelerationStructure = acceleration_structure.handle;
    acceleration_structure_build_geometry_info.scratchData.deviceAddress = scratch_buffer->getBufferDeviceAddress();

    VkAccelerationStructureBuildRangeInfoKHR top_level_acceleration_structure_build_range_info =
    {
        .primitiveCount = static_cast<uint32_t>(blas_instances.size()),
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0
    };

    cmdBuildTlas(acceleration_structure_build_geometry_info, &top_level_acceleration_structure_build_range_info);
}
