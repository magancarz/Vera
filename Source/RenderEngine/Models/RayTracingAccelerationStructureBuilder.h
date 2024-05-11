#pragma once

#include <vector>

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan.hpp>

#include "RenderEngine/RenderingAPI/Device.h"
#include "World/World.h"
#include "RenderEngine/Models/AccelerationStructure.h"

class RayTracingAccelerationStructureBuilder
{
public:
    struct BlasInput
    {
        std::vector<VkAccelerationStructureGeometryKHR> acceleration_structure_geometry;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> acceleration_structure_build_offset_info;
        VkBuildAccelerationStructureFlagsKHR flags{0};
    };

    RayTracingAccelerationStructureBuilder(Device& device);

    AccelerationStructure buildBottomLevelAccelerationStructure(const BlasInput& blas_input);
    std::vector<AccelerationStructure> buildBottomLevelAccelerationStructures(
            const std::vector<BlasInput>& blas_input,
            VkBuildAccelerationStructureFlagsKHR flags);
    AccelerationStructure buildTopLevelAccelerationStructure(const std::vector<BlasInstance*>& blas_instances);

private:
    struct BuildAccelerationStructure
    {
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo;
        AccelerationStructure as;
        AccelerationStructure cleanupAS;
    };

    void cmdCreateBlas(
            VkCommandBuffer command_buffer,
            std::vector<uint32_t>& indices,
            std::vector<BuildAccelerationStructure>& build_as,
            VkDeviceAddress scratch_address,
            VkQueryPool query_pool);
    void cmdCompactBlas(
            VkCommandBuffer command_buffer,
            std::vector<uint32_t>& indices,
            std::vector<BuildAccelerationStructure>& build_as,
            VkQueryPool query_pool);
    void destroyNonCompacted(
            std::vector<uint32_t>& indices,
            std::vector<BuildAccelerationStructure>& build_as);

    Device& device;
};
