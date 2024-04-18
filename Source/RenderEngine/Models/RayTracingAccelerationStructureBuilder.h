#pragma once

#include <vector>

#include <vulkan/vulkan_core.h>
#include "RenderEngine/RenderingAPI/Device.h"
#include "World/World.h"
#include "RenderEngine/Models/AccelerationStructure.h"

class RayTracingAccelerationStructureBuilder
{
public:
    struct BlasInput
    {
        VkAccelerationStructureGeometryKHR acceleration_structure_geometry;
        VkAccelerationStructureBuildRangeInfoKHR acceleration_structure_build_offset_info;
        VkBuildAccelerationStructureFlagsKHR flags{0};
    };

    RayTracingAccelerationStructureBuilder(Device& device);

    AccelerationStructure buildBottomLevelAccelerationStructure(const BlasInput& blas_input);
    AccelerationStructure buildTopLevelAccelerationStructure(const std::vector<BlasInstance*>& blas_instances);

private:

    Device& device;
};
