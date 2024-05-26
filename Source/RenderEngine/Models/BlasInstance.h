#pragma once

#include "RenderEngine/Memory/Buffer.h"

struct BlasInstance
{
    VkAccelerationStructureInstanceKHR bottomLevelAccelerationStructureInstance{};
    std::unique_ptr<Buffer> bottom_level_geometry_instance_buffer{};
    VkDeviceAddress bottomLevelGeometryInstanceDeviceAddress{};
};