#pragma once

#include <memory>

#include "Memory/Buffer.h"

struct BlasInstance
{
    VkAccelerationStructureInstanceKHR bottom_level_acceleration_structure_instance{};
    std::unique_ptr<Buffer> bottom_level_geometry_instance_buffer{};
    VkDeviceAddress bottom_level_geometry_instance_device_address{};
};