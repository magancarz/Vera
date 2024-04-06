#pragma once

#include <vector>

#include <vulkan/vulkan_core.h>
#include "RenderEngine/RenderingAPI/Device.h"

class RayTracingBuilder
{
public:
    struct BlasInput
    {
//        std::vector<VkAccelerationStructureGeometryKHR> acceleration_structure_geometry;
//        std::vector<VkAccelerationStructureBuildRangeInfoKHR> acceleration_structure_build_offset_info;
        VkAccelerationStructureGeometryKHR acceleration_structure_geometry;
        VkAccelerationStructureBuildRangeInfoKHR acceleration_structure_build_offset_info;
        VkBuildAccelerationStructureFlagsKHR flags{0};
    };

    void setup(uint32_t in_queue_index, VkPhysicalDeviceRayTracingPipelinePropertiesKHR in_ray_tracing_properties);

    void buildBlas(
        Device& device,
        const BlasInput& input);

    void buildTlas(Device& device);

private:
    uint32_t queue_index{0};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    VkDeviceAddress bottomLevelAccelerationStructureDeviceAddress;
};
