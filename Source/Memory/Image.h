#pragma once
#include "AllocatorInfo.h"

class Image
{
public:
    Image(VkImage image, const VulkanMemoryAllocatorInfo& allocator_info);
    ~Image();

    VkImage getImage() { return image; }
    VulkanMemoryAllocatorInfo getAllocatorInfo() { return allocator_info; }

private:
    VkImage image{VK_NULL_HANDLE};
    VulkanMemoryAllocatorInfo allocator_info{};
};
