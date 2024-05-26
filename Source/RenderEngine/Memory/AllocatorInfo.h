#pragma once

#include "RenderEngine/Memory/Vulkan/VmaUsage.h"

struct VulkanMemoryAllocatorInfo
{
    VmaAllocator vma_allocator;
    VmaAllocation vma_allocation;
    VmaAllocationInfo vma_allocation_info;
};
