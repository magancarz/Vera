#include "RayTracingAccelerationStructureBuilder.h"

#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

RayTracingAccelerationStructureBuilder::RayTracingAccelerationStructureBuilder(VulkanFacade& device)
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
    acceleration_structure_build_geometry_info.pGeometries = &blas_input.acceleration_structure_geometry[0];

    std::vector<uint32_t> bottom_level_max_primitive_count_list = {blas_input.acceleration_structure_build_offset_info[0].primitiveCount};

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
    bottom_level_acceleration_structure_create_info.buffer = acceleration_structure.acceleration_structure_buffer->getBuffer();
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
    bottom_level_acceleration_structure_build_range_info.primitiveCount = blas_input.acceleration_structure_build_offset_info[0].primitiveCount;
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

std::vector<AccelerationStructure> RayTracingAccelerationStructureBuilder::buildBottomLevelAccelerationStructures(
        const std::vector<BlasInput>& blas_input,
        VkBuildAccelerationStructureFlagsKHR flags)
{
    uint32_t blas_count = blas_input.size();
    VkDeviceSize as_total_size{0};
    uint32_t number_of_compactions{0};
    VkDeviceSize max_scratch_buffer_size{0};

    std::vector<BuildAccelerationStructure> build_as(blas_count);
    for (uint32_t idx = 0; idx < blas_count; ++idx)
    {
        build_as[idx].buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_as[idx].buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_as[idx].buildInfo.flags = blas_input[idx].flags | flags;
        build_as[idx].buildInfo.geometryCount = static_cast<uint32_t>(blas_input[idx].acceleration_structure_geometry.size());
        build_as[idx].buildInfo.pGeometries = blas_input[idx].acceleration_structure_geometry.data();

        build_as[idx].rangeInfo = blas_input[idx].acceleration_structure_build_offset_info.data();

        std::vector<uint32_t> max_primitives_count(blas_input[idx].acceleration_structure_build_offset_info.size());
        for (uint32_t tt = 0; tt < blas_input[idx].acceleration_structure_build_offset_info.size(); ++tt)
        {
            max_primitives_count[tt] = blas_input[idx].acceleration_structure_build_offset_info[tt].primitiveCount;
        }

        pvkGetAccelerationStructureBuildSizesKHR(
                device.getDevice(),
                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                &build_as[idx].buildInfo,
                max_primitives_count.data(),
                &build_as[idx].sizeInfo);

        as_total_size += build_as[idx].sizeInfo.accelerationStructureSize;
        max_scratch_buffer_size = std::max(max_scratch_buffer_size, build_as[idx].sizeInfo.buildScratchSize);
        number_of_compactions += build_as[idx].buildInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR ? 1 : 0;
    }

    auto scratch_buffer = std::make_unique<Buffer>
    (
            device,
            max_scratch_buffer_size,
            1,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    VkDeviceAddress scratch_buffer_device_address = scratch_buffer->getBufferDeviceAddress();

    VkQueryPool query_pool{VK_NULL_HANDLE};
    if (number_of_compactions > 0)
    {
        assert(number_of_compactions == blas_count && "Mix of on/off compaction is not allowed");
        VkQueryPoolCreateInfo query_pool_create_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_pool_create_info.queryCount = blas_count;
        query_pool_create_info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        vkCreateQueryPool(device.getDevice(), &query_pool_create_info, VulkanDefines::NO_CALLBACK, &query_pool);
    }

    std::vector<uint32_t> indices;
    VkDeviceSize batch_size{0};
    VkDeviceSize batch_limit{256'000'000};
    for (uint32_t idx = 0; idx < blas_count; ++idx)
    {
        indices.push_back(idx);
        batch_size += build_as[idx].sizeInfo.accelerationStructureSize;

        if (batch_size >= batch_limit || idx == blas_count - 1)
        {
            {
                VkCommandBuffer command_buffer = device.beginSingleTimeCommands();
                cmdCreateBlas(command_buffer, indices, build_as, scratch_buffer_device_address, query_pool);
                device.endSingleTimeCommands(command_buffer);
            }

            if (query_pool)
            {
                VkCommandBuffer command_buffer = device.beginSingleTimeCommands();
                cmdCompactBlas(command_buffer, indices, build_as, query_pool);
                device.endSingleTimeCommands(command_buffer);
                destroyNonCompacted(indices, build_as);
            }

            batch_size = 0;
            indices.clear();
        }
    }

    vkDestroyQueryPool(device.getDevice(), query_pool, VulkanDefines::NO_CALLBACK);

    std::vector<AccelerationStructure> out_blas_values;
    for (auto& blas : build_as)
    {
        out_blas_values.emplace_back(std::move(blas.as));
    }

    return out_blas_values;
}

void RayTracingAccelerationStructureBuilder::cmdCreateBlas(
        VkCommandBuffer command_buffer,
        std::vector<uint32_t>& indices,
        std::vector<BuildAccelerationStructure>& build_as,
        VkDeviceAddress scratch_address,
        VkQueryPool query_pool)
{
    if (query_pool)
    {
        vkResetQueryPool(device.getDevice(), query_pool, 0, static_cast<uint32_t>(indices.size()));
    }

    uint32_t query_count{0};
    for (const auto& idx : indices)
    {
        VkAccelerationStructureCreateInfoKHR create_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        create_info.size = build_as[idx].sizeInfo.accelerationStructureSize;
        build_as[idx].as.acceleration_structure_buffer = std::make_unique<Buffer>
        (
                device,
                create_info.size,
                1,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        build_as[idx].as.bottom_level_acceleration_structure_device_address = build_as[idx].as.acceleration_structure_buffer->getBufferDeviceAddress();

        create_info.buffer = build_as[idx].as.acceleration_structure_buffer->getBuffer();
        if (pvkCreateAccelerationStructureKHR(
                device.getDevice(),
                &create_info,
                VulkanDefines::NO_CALLBACK,
                &build_as[idx].as.acceleration_structure) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateAccelerationStructureKHR");
        }

        build_as[idx].buildInfo.dstAccelerationStructure = build_as[idx].as.acceleration_structure;
        build_as[idx].buildInfo.scratchData.deviceAddress = scratch_address;

        pvkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &build_as[idx].buildInfo, &build_as[idx].rangeInfo);

        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(
                command_buffer,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                0,
                1,
                &barrier,
                0,
                nullptr,
                0,
                nullptr);

        if (query_pool)
        {
            pvkCmdWriteAccelerationStructuresPropertiesKHR(
                    command_buffer,
                    1,
                    &build_as[idx].buildInfo.dstAccelerationStructure,
                    VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                    query_pool,
                    query_count);
        }
    }
}

void RayTracingAccelerationStructureBuilder::cmdCompactBlas(
        VkCommandBuffer command_buffer,
        std::vector<uint32_t>& indices,
        std::vector<BuildAccelerationStructure>& build_as,
        VkQueryPool query_pool)
{
    uint32_t query_count{0};
    std::vector<AccelerationStructure> cleanup_as;

    std::vector<VkDeviceSize> compact_sizes(static_cast<uint32_t>(indices.size()));
    vkGetQueryPoolResults(
            device.getDevice(),
            query_pool,
            0,
            static_cast<uint32_t>(compact_sizes.size()),
            compact_sizes.size() * sizeof(VkDeviceSize),
            compact_sizes.data(),
            sizeof(VkDeviceSize),
            VK_QUERY_RESULT_WAIT_BIT);

    for (uint32_t idx : indices)
    {
        build_as[idx].cleanupAS = std::move(build_as[idx].as);
        build_as[idx].sizeInfo.accelerationStructureSize = compact_sizes[query_count++];

        VkAccelerationStructureCreateInfoKHR create_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        create_info.size = build_as[idx].sizeInfo.accelerationStructureSize;
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_as[idx].as.acceleration_structure_buffer = std::make_unique<Buffer>
        (
                device,
                create_info.size,
                1,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        build_as[idx].as.bottom_level_acceleration_structure_device_address = build_as[idx].as.acceleration_structure_buffer->getBufferDeviceAddress();

        create_info.buffer = build_as[idx].as.acceleration_structure_buffer->getBuffer();
        if (pvkCreateAccelerationStructureKHR(
                device.getDevice(),
                &create_info,
                VulkanDefines::NO_CALLBACK,
                &build_as[idx].as.acceleration_structure) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateAccelerationStructureKHR");
        }

        VkCopyAccelerationStructureInfoKHR copy_info{VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
        copy_info.src  = build_as[idx].buildInfo.dstAccelerationStructure;
        copy_info.dst  = build_as[idx].as.acceleration_structure;
        copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
        pvkCmdCopyAccelerationStructureKHR(command_buffer, &copy_info);
    }
}

void RayTracingAccelerationStructureBuilder::destroyNonCompacted(
        std::vector<uint32_t>& indices,
        std::vector<BuildAccelerationStructure>& build_as)
{
    for (uint32_t idx : indices)
    {
        pvkDestroyAccelerationStructureKHR(device.getDevice(), build_as[idx].cleanupAS.acceleration_structure, VulkanDefines::NO_CALLBACK);
    }
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