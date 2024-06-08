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

private:
    struct BuildAccelerationStructure
    {
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo;
        AccelerationStructure as;
        AccelerationStructure cleanupAS;
    };

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
