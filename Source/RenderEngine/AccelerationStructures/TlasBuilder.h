#pragma once

#include <vector>

#include "AccelerationStructure.h"

class MemoryAllocator;
struct BlasInstance;

class TlasBuilder
{
public:
    static AccelerationStructure buildTopLevelAccelerationStructure(
        VulkanHandler& device, MemoryAllocator& memory_allocator, const std::vector<BlasInstance>& blas_instances);
};
