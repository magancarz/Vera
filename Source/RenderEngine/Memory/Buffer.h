#pragma once

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "BufferInfo.h"
#include "AllocatorInfo.h"

class Buffer {
public:
    Buffer(VulkanFacade& vulkan_facade, const VulkanMemoryAllocatorInfo& allocator_info, VkBuffer buffer, uint32_t buffer_size);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkResult map();
    void unmap();

    void writeToBuffer(void* data, VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
    void copyFromBuffer(const std::unique_ptr<Buffer>& src_buffer);
    [[nodiscard]] VkResult flush(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0) const;
    VkDescriptorBufferInfo descriptorInfo(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);

    [[nodiscard]] VkBuffer getBuffer() const { return buffer; }
    [[nodiscard]] uint32_t getSize() const { return buffer_size; }
    [[nodiscard]] void* getMappedMemory() const { return mapped; }
    VkDeviceAddress getBufferDeviceAddress();

private:
    VulkanFacade& vulkan_facade;
    VulkanMemoryAllocatorInfo allocator_info;
    VkBuffer buffer;
    uint32_t buffer_size;

    void* mapped = nullptr;
};