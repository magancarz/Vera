#pragma once

#include <vector>

#include "RenderEngine/RenderingAPI/VulkanHandler.h"
#include "RenderEngine/AccelerationStructures/AccelerationStructure.h"

class MemoryAllocator;

class BlasBuilder
{
public:
    struct BlasInput
    {
        std::vector<VkAccelerationStructureGeometryKHR> acceleration_structure_geometry;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> acceleration_structure_build_offset_info;
        VkBuildAccelerationStructureFlagsKHR flags{0};
    };

    static std::vector<AccelerationStructure> buildBottomLevelAccelerationStructures(
        VulkanHandler& device, MemoryAllocator& memory_allocator,
        const std::vector<BlasInput>& blas_input,
        VkBuildAccelerationStructureFlagsKHR flags);

    static void updateBottomLevelAccelerationStructures(
        VulkanHandler& device, MemoryAllocator& memory_allocator,
        const std::vector<VkAccelerationStructureKHR>& source_acceleration_structures,
        const std::vector<BlasInput>& blas_input,
        VkBuildAccelerationStructureFlagsKHR flags);

private:
    struct BuildAccelerationStructure
    {
        VkAccelerationStructureBuildGeometryInfoKHR build_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        VkAccelerationStructureBuildSizesInfoKHR size_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        const VkAccelerationStructureBuildRangeInfoKHR* range_info;
        AccelerationStructure as;
        AccelerationStructure cleanup_as;
    };

    static std::vector<BuildAccelerationStructure> fillBuildInfos(
        const Device& logical_device,
        const std::vector<BlasInput>& blas_input,
        VkBuildAccelerationStructureModeKHR mode,
        VkBuildAccelerationStructureFlagsKHR flags,
        VkDeviceSize& as_total_size,
        uint32_t& number_of_compactions,
        VkDeviceSize& max_scratch_buffer_size);
    static std::unique_ptr<Buffer> createScratchBuffer(MemoryAllocator& memory_allocator, uint32_t scratch_buffer_size);

    static void cmdCreateBlas(
        VulkanHandler& device, MemoryAllocator& memory_allocator,
        VkCommandBuffer command_buffer,
        std::vector<uint32_t>& indices,
        std::vector<BuildAccelerationStructure>& build_as,
        VkDeviceAddress scratch_address,
        VkQueryPool query_pool);
    static void cmdCompactBlas(
        VulkanHandler& device, MemoryAllocator& memory_allocator,
        VkCommandBuffer command_buffer,
        std::vector<uint32_t>& indices,
        std::vector<BuildAccelerationStructure>& build_as,
        VkQueryPool query_pool);
    static void destroyNonCompacted(
        VulkanHandler& device,
        std::vector<uint32_t>& indices,
        std::vector<BuildAccelerationStructure>& build_as);
};
