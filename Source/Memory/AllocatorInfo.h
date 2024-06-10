#pragma once

#include "Vulkan/VmaUsage.h"

struct VulkanMemoryAllocatorInfo
{
    VmaAllocator vma_allocator;
    VmaAllocation vma_allocation;
    VmaAllocationInfo vma_allocation_info;
};
