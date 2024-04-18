#pragma once

#include "RenderEngine/RenderingAPI/Buffer.h"

struct BlasInstance
{
    VkAccelerationStructureInstanceKHR bottomLevelAccelerationStructureInstance{};
    std::unique_ptr<Buffer> bottom_level_geometry_instance_buffer{};
    VkDeviceAddress bottomLevelGeometryInstanceDeviceAddress{};
};