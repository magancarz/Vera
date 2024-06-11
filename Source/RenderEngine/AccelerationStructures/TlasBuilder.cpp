#include "TlasBuilder.h"

#include <RenderEngine/RenderingAPI/VulkanDefines.h>

#include "Memory/MemoryAllocator.h"
#include "BlasInstance.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

AccelerationStructure TlasBuilder::buildTopLevelAccelerationStructure(
    VulkanHandler& device, MemoryAllocator& memory_allocator, const std::vector<BlasInstance>& blas_instances)
{
    AccelerationStructure tlas{};

    std::vector<VkAccelerationStructureInstanceKHR> instances_temp{};
    instances_temp.reserve(blas_instances.size());
    std::ranges::transform(blas_instances.begin(), blas_instances.end(), std::back_inserter(instances_temp),
                           [](const BlasInstance& blas_instance)
                           {
                               return blas_instance.bottom_level_acceleration_structure_instance;
                           });

    auto instances_staging_buffer = memory_allocator.createStagingBuffer(
        sizeof(VkAccelerationStructureInstanceKHR), instances_temp.size(), instances_temp.data());

    BufferInfo instances_buffer_info{};
    instances_buffer_info.instance_size = sizeof(VkAccelerationStructureInstanceKHR);
    instances_buffer_info.instance_count = static_cast<uint32_t>(instances_temp.size());
    instances_buffer_info.usage_flags =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    instances_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    auto instances_buffer = memory_allocator.createBuffer(instances_buffer_info);
    instances_buffer->copyFrom(*instances_staging_buffer);

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
    top_level_acceleration_structure_build_geometry_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
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

    std::vector<uint32_t> top_level_max_primitive_count_list{static_cast<uint32_t>(blas_instances.size())};

    pvkGetAccelerationStructureBuildSizesKHR(
        device.getDeviceHandle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &top_level_acceleration_structure_build_geometry_info,
        top_level_max_primitive_count_list.data(),
        &top_level_acceleration_structure_build_sizes_info);

    BufferInfo acceleration_structure_buffer_info{};
    acceleration_structure_buffer_info.instance_size = static_cast<uint32_t>(top_level_acceleration_structure_build_sizes_info.accelerationStructureSize);
    acceleration_structure_buffer_info.instance_count = 1;
    acceleration_structure_buffer_info.usage_flags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    acceleration_structure_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    auto acceleration_structure_buffer = memory_allocator.createBuffer(acceleration_structure_buffer_info);

    VkAccelerationStructureCreateInfoKHR top_level_acceleration_structure_create_info{};
    top_level_acceleration_structure_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    top_level_acceleration_structure_create_info.createFlags = 0;
    top_level_acceleration_structure_create_info.buffer = acceleration_structure_buffer->getBuffer();
    top_level_acceleration_structure_create_info.offset = 0;
    top_level_acceleration_structure_create_info.size = top_level_acceleration_structure_build_sizes_info.accelerationStructureSize;
    top_level_acceleration_structure_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    top_level_acceleration_structure_create_info.deviceAddress = 0;

    VkAccelerationStructureKHR acceleration_structure;
    if (pvkCreateAccelerationStructureKHR(
        device.getDeviceHandle(),
        &top_level_acceleration_structure_create_info,
        VulkanDefines::NO_CALLBACK,
        &acceleration_structure) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateAccelerationStructureKHR");
    }

    BufferInfo top_level_acceleration_structure_scratch_buffer_info{};
    top_level_acceleration_structure_scratch_buffer_info.instance_size =
        static_cast<uint32_t>(top_level_acceleration_structure_build_sizes_info.buildScratchSize);
    top_level_acceleration_structure_scratch_buffer_info.instance_count = 1;
    top_level_acceleration_structure_scratch_buffer_info.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    top_level_acceleration_structure_scratch_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    auto top_level_acceleration_structure_scratch_buffer = memory_allocator.createBuffer(top_level_acceleration_structure_scratch_buffer_info);

    VkDeviceAddress top_level_acceleration_structure_scratch_buffer_device_address =
        top_level_acceleration_structure_scratch_buffer->getBufferDeviceAddress();

    top_level_acceleration_structure_build_geometry_info.dstAccelerationStructure = acceleration_structure;
    top_level_acceleration_structure_build_geometry_info.scratchData.deviceAddress =
        top_level_acceleration_structure_scratch_buffer_device_address;

    VkAccelerationStructureBuildRangeInfoKHR top_level_acceleration_structure_build_range_info =
    {
        .primitiveCount = static_cast<uint32_t>(blas_instances.size()),
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0
    };

    const VkAccelerationStructureBuildRangeInfoKHR* top_level_acceleration_structure_build_range_infos = &
        top_level_acceleration_structure_build_range_info;

    VkCommandBuffer command_buffer = device.getCommandPool().beginSingleTimeCommands();

    pvkCmdBuildAccelerationStructuresKHR(
        command_buffer, 1,
        &top_level_acceleration_structure_build_geometry_info,
        &top_level_acceleration_structure_build_range_infos);

    device.getCommandPool().endSingleTimeCommands(command_buffer);

    return AccelerationStructure
    {
        .acceleration_structure = acceleration_structure,
        .acceleration_structure_buffer = std::move(acceleration_structure_buffer)
    };
}
