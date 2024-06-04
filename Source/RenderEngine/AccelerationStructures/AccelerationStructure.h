#pragma once

#include "Memory/Buffer.h"

struct AccelerationStructure
{
    VkAccelerationStructureKHR acceleration_structure;
    std::unique_ptr<Buffer> acceleration_structure_buffer;
    VkDeviceAddress bottom_level_acceleration_structure_device_address;
};