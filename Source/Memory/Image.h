#pragma once
#include "AllocatorInfo.h"

class Image
{
public:
    Image(VkImage image, const VulkanMemoryAllocatorInfo& allocator_info);
    ~Image();

    Image(const Image&) = delete;
    Image operator=(const Image&) = delete;
    Image(const Image&&) = delete;
    Image operator=(const Image&&) = delete;

    [[nodiscard]] VkImage getImage() const { return image; }
    [[nodiscard]] VulkanMemoryAllocatorInfo getAllocatorInfo() const { return allocator_info; }

private:
    VkImage image{VK_NULL_HANDLE};
    VulkanMemoryAllocatorInfo allocator_info{};
};
