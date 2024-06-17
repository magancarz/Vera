#include "BlasBuilder.h"

#include "Blas.h"
#include "BlasInstance.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "Memory/MemoryAllocator.h"

std::vector<AccelerationStructure> BlasBuilder::buildBottomLevelAccelerationStructures(
        VulkanHandler& device, MemoryAllocator& memory_allocator,
        const std::vector<BlasInput>& blas_input,
        VkBuildAccelerationStructureFlagsKHR flags)
{
    uint32_t blas_count = blas_input.size();
    VkDeviceSize as_total_size{0};
    uint32_t number_of_compactions{0};
    VkDeviceSize max_scratch_buffer_size{0};

    std::vector<BuildAccelerationStructure> build_as = fillBuildInfos(
        device.getLogicalDevice(),
        blas_input,
        VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        flags,
        as_total_size,
        number_of_compactions,
        max_scratch_buffer_size);

    auto scratch_buffer = createScratchBuffer(memory_allocator, max_scratch_buffer_size);

    VkQueryPool query_pool{VK_NULL_HANDLE};
    if (number_of_compactions > 0)
    {
        assert(number_of_compactions == blas_count && "Mix of on/off compaction is not allowed");
        VkQueryPoolCreateInfo query_pool_create_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_pool_create_info.queryCount = blas_count;
        query_pool_create_info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        vkCreateQueryPool(device.getDeviceHandle(), &query_pool_create_info, VulkanDefines::NO_CALLBACK, &query_pool);
    }

    std::vector<uint32_t> indices;
    VkDeviceSize batch_size{0};
    VkDeviceSize batch_limit{256'000'000};
    for (uint32_t idx = 0; idx < blas_count; ++idx)
    {
        indices.push_back(idx);
        batch_size += build_as[idx].size_info.accelerationStructureSize;

        if (batch_size >= batch_limit || idx == blas_count - 1)
        {
            VkCommandBuffer create_command_buffer = device.getCommandPool().beginSingleTimeCommands();
            cmdCreateBlas(device, memory_allocator, create_command_buffer, indices, build_as, scratch_buffer->getBufferDeviceAddress(), query_pool);
            device.getCommandPool().endSingleTimeCommands(create_command_buffer);

            if (query_pool)
            {
                VkCommandBuffer compact_command_buffer = device.getCommandPool().beginSingleTimeCommands();
                cmdCompactBlas(device, memory_allocator, compact_command_buffer, indices, build_as, query_pool);
                device.getCommandPool().endSingleTimeCommands(compact_command_buffer);
                destroyNonCompacted(device, indices, build_as);
            }

            batch_size = 0;
            indices.clear();
        }
    }

    vkDestroyQueryPool(device.getDeviceHandle(), query_pool, VulkanDefines::NO_CALLBACK);

    std::vector<AccelerationStructure> out_blas_values;
    for (auto& blas : build_as)
    {
        out_blas_values.emplace_back(std::move(blas.as));
    }

    return out_blas_values;
}

std::vector<BlasBuilder::BuildAccelerationStructure> BlasBuilder::fillBuildInfos(
        const Device& logical_device,
        const std::vector<BlasInput>& blas_input,
        VkBuildAccelerationStructureModeKHR mode,
        VkBuildAccelerationStructureFlagsKHR flags,
        VkDeviceSize& as_total_size,
        uint32_t& number_of_compactions,
        VkDeviceSize& max_scratch_buffer_size)
{
    std::vector<BuildAccelerationStructure> build_as(blas_input.size());
    for (size_t idx = 0; idx < blas_input.size(); ++idx)
    {
        build_as[idx].build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_as[idx].build_info.mode = mode;
        build_as[idx].build_info.flags = blas_input[idx].flags | flags;
        build_as[idx].build_info.geometryCount = static_cast<uint32_t>(blas_input[idx].acceleration_structure_geometry.size());
        build_as[idx].build_info.pGeometries = blas_input[idx].acceleration_structure_geometry.data();

        build_as[idx].range_info = blas_input[idx].acceleration_structure_build_offset_info.data();

        std::vector<uint32_t> max_primitives_count(blas_input[idx].acceleration_structure_build_offset_info.size());
        for (uint32_t tt = 0; tt < blas_input[idx].acceleration_structure_build_offset_info.size(); ++tt)
        {
            max_primitives_count[tt] = blas_input[idx].acceleration_structure_build_offset_info[tt].primitiveCount;
        }

        pvkGetAccelerationStructureBuildSizesKHR(
            logical_device.getDevice(),
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &build_as[idx].build_info,
            max_primitives_count.data(),
            &build_as[idx].size_info);

        as_total_size += build_as[idx].size_info.accelerationStructureSize;
        max_scratch_buffer_size = std::max(max_scratch_buffer_size, build_as[idx].size_info.buildScratchSize);
        number_of_compactions += build_as[idx].build_info.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR ? 1 : 0;
    }

    return build_as;
}

std::unique_ptr<Buffer> BlasBuilder::createScratchBuffer(MemoryAllocator& memory_allocator, uint32_t scratch_buffer_size)
{
    BufferInfo scratch_buffer_info{};
    scratch_buffer_info.instance_size = scratch_buffer_size;
    scratch_buffer_info.instance_count = 1;
    scratch_buffer_info.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    scratch_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    return memory_allocator.createBuffer(scratch_buffer_info);
}

void BlasBuilder::cmdCreateBlas(
    VulkanHandler& device, MemoryAllocator& memory_allocator,
    VkCommandBuffer command_buffer,
    std::vector<uint32_t>& indices,
    std::vector<BuildAccelerationStructure>& build_as,
    VkDeviceAddress scratch_address,
    VkQueryPool query_pool)
{
    if (query_pool)
    {
        vkResetQueryPool(device.getDeviceHandle(), query_pool, 0, static_cast<uint32_t>(indices.size()));
    }

    uint32_t query_count{0};
    for (const auto& idx : indices)
    {
        VkAccelerationStructureCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        create_info.size = build_as[idx].size_info.accelerationStructureSize;

        BufferInfo acceleration_structure_buffer_info{};
        acceleration_structure_buffer_info.instance_size = static_cast<uint32_t>(create_info.size);
        acceleration_structure_buffer_info.instance_count = 1;
        acceleration_structure_buffer_info.usage_flags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        acceleration_structure_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        build_as[idx].as.buffer = memory_allocator.createBuffer(acceleration_structure_buffer_info);

        create_info.buffer = build_as[idx].as.buffer->getBuffer();
        if (pvkCreateAccelerationStructureKHR(
                device.getDeviceHandle(),
                &create_info,
                VulkanDefines::NO_CALLBACK,
                &build_as[idx].as.handle) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create bottom level acceleration structure!");
        }

        build_as[idx].build_info.dstAccelerationStructure = build_as[idx].as.handle;
        build_as[idx].build_info.scratchData.deviceAddress = scratch_address;

        pvkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &build_as[idx].build_info, &build_as[idx].range_info);

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
                &build_as[idx].build_info.dstAccelerationStructure,
                VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                query_pool,
                query_count);
        }
    }
}

void BlasBuilder::cmdCompactBlas(
    VulkanHandler& device, MemoryAllocator& memory_allocator,
    VkCommandBuffer command_buffer,
    std::vector<uint32_t>& indices,
    std::vector<BuildAccelerationStructure>& build_as,
    VkQueryPool query_pool)
{
    uint32_t query_count{0};
    std::vector<AccelerationStructure> cleanup_as;

    std::vector<VkDeviceSize> compact_sizes(static_cast<uint32_t>(indices.size()));
    vkGetQueryPoolResults(
        device.getDeviceHandle(),
        query_pool,
        0,
        static_cast<uint32_t>(compact_sizes.size()),
        compact_sizes.size() * sizeof(VkDeviceSize),
        compact_sizes.data(),
        sizeof(VkDeviceSize),
        VK_QUERY_RESULT_WAIT_BIT);

    for (uint32_t idx : indices)
    {
        build_as[idx].cleanup_as = std::move(build_as[idx].as);
        build_as[idx].size_info.accelerationStructureSize = compact_sizes[query_count++];

        VkAccelerationStructureCreateInfoKHR create_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        create_info.size = build_as[idx].size_info.accelerationStructureSize;
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        BufferInfo acceleration_structure_buffer_info{};
        acceleration_structure_buffer_info.instance_size = static_cast<uint32_t>(create_info.size);
        acceleration_structure_buffer_info.instance_count = 1;
        acceleration_structure_buffer_info.usage_flags =
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        acceleration_structure_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        build_as[idx].as.buffer = memory_allocator.createBuffer(acceleration_structure_buffer_info);

        create_info.buffer = build_as[idx].as.buffer->getBuffer();
        if (pvkCreateAccelerationStructureKHR(
            device.getDeviceHandle(),
            &create_info,
            VulkanDefines::NO_CALLBACK,
            &build_as[idx].as.handle) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateAccelerationStructureKHR");
        }

        VkCopyAccelerationStructureInfoKHR copy_info{VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
        copy_info.src = build_as[idx].build_info.dstAccelerationStructure;
        copy_info.dst = build_as[idx].as.handle;
        copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
        pvkCmdCopyAccelerationStructureKHR(command_buffer, &copy_info);
    }
}

void BlasBuilder::destroyNonCompacted(
    VulkanHandler& device,
    std::vector<uint32_t>& indices,
    std::vector<BuildAccelerationStructure>& build_as)
{
    for (uint32_t idx : indices)
    {
        pvkDestroyAccelerationStructureKHR(device.getDeviceHandle(), build_as[idx].cleanup_as.handle, VulkanDefines::NO_CALLBACK);
    }
}

void BlasBuilder::updateBottomLevelAccelerationStructures(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        const std::vector<VkAccelerationStructureKHR>& source_acceleration_structures,
        const std::vector<BlasInput>& blas_input,
        VkBuildAccelerationStructureFlagsKHR flags)
{
    VkDeviceSize as_total_size{0};
    uint32_t number_of_compactions{0};
    VkDeviceSize max_scratch_buffer_size{0};
    std::vector<BuildAccelerationStructure> build_infos = fillBuildInfos(
        device.getLogicalDevice(),
        blas_input,
        VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR,
        flags,
        as_total_size,
        number_of_compactions,
        max_scratch_buffer_size);

    std::unique_ptr<Buffer> scratch_buffer = createScratchBuffer(memory_allocator, max_scratch_buffer_size);

    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> final_build_infos(build_infos.size());
    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> final_range_infos(build_infos.size());
    for (size_t i = 0; i < build_infos.size(); ++i)
    {
        build_infos[i].build_info.srcAccelerationStructure = source_acceleration_structures[i];
        build_infos[i].build_info.dstAccelerationStructure = source_acceleration_structures[i];
        build_infos[i].build_info.scratchData.deviceAddress = scratch_buffer->getBufferDeviceAddress();

        final_build_infos[i] = build_infos[i].build_info;
        final_range_infos[i] = build_infos[i].range_info;
    }

    // TODO: update BLASes in batches
    VkCommandBuffer command_buffer = device.getCommandPool().beginSingleTimeCommands();
    pvkCmdBuildAccelerationStructuresKHR(command_buffer, build_infos.size(), final_build_infos.data(), final_range_infos.data());
    device.getCommandPool().endSingleTimeCommands(command_buffer);
}
