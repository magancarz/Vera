#pragma once

#include <memory>

#include "Memory/Buffer.h"

struct AccelerationStructure
{
    VkAccelerationStructureKHR handle;
    std::unique_ptr<Buffer> buffer;
};