#pragma once

#include "Memory/Vulkan/VmaUsage.h"
#include "Memory/MemoryAllocator.h"

class VulkanMemoryAllocator : public MemoryAllocator
{
public:
    VulkanMemoryAllocator(VulkanHandler& vulkan_handler);
    ~VulkanMemoryAllocator() override;

    VulkanMemoryAllocator(const VulkanMemoryAllocator&) = delete;
    VulkanMemoryAllocator& operator=(const VulkanMemoryAllocator&) = delete;
    VulkanMemoryAllocator(VulkanMemoryAllocator&&) = default;
    VulkanMemoryAllocator& operator=(VulkanMemoryAllocator&&) = delete;

    std::unique_ptr<Buffer> createBuffer(const BufferInfo& buffer_info) override;
    std::unique_ptr<Buffer> createStagingBuffer(uint32_t instance_size, uint32_t instance_count) override;
    std::unique_ptr<Buffer> createStagingBuffer(uint32_t instance_size, uint32_t instance_count, const void *data) override;
    std::unique_ptr<Image> createImage(const VkImageCreateInfo& image_create_info) override;

private:
    VulkanHandler& vulkan_handler;

    void initializeVMA();

    VmaAllocator allocator{VK_NULL_HANDLE};

    static VkDeviceSize getAlignment(VkDeviceSize instance_size, VkDeviceSize min_offset_alignment);
    BufferInfo createStagingBufferInfo(uint32_t instance_size, uint32_t instance_count);
};
