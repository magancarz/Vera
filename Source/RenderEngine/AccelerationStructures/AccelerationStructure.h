#pragma once

#include <memory>

#include "Memory/Buffer.h"

struct AccelerationStructure
{
    VkAccelerationStructureKHR acceleration_structure;
    std::unique_ptr<Buffer> acceleration_structure_buffer;
};