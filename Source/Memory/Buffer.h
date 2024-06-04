#pragma once

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "AllocatorInfo.h"

class Buffer {
public:
    Buffer(VulkanFacade& vulkan_facade, const VulkanMemoryAllocatorInfo& allocator_info, VkBuffer buffer, uint32_t buffer_size);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkResult map();
    void unmap();

    void writeToBuffer(const void* data);
    void copyFromBuffer(const std::unique_ptr<Buffer>& src_buffer);
    [[nodiscard]] VkResult flush() const;
    VkDescriptorBufferInfo descriptorInfo();

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