#pragma once

#include "RenderEngine/RenderingAPI/VulkanHandler.h"
#include "AllocatorInfo.h"

class Buffer {
public:
    Buffer(
        Device& logical_device,
        CommandPool& command_pool,
        const VulkanMemoryAllocatorInfo& allocator_info,
        VkBuffer buffer,
        uint32_t buffer_size);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkResult map();
    void unmap();

    void writeToBuffer(const void* data);
    void copyFrom(const Buffer& src_buffer) const;
    [[nodiscard]] VkDescriptorBufferInfo descriptorInfo() const;

    [[nodiscard]] VkBuffer getBuffer() const { return buffer; }
    [[nodiscard]] uint32_t getSize() const { return buffer_size; }
    [[nodiscard]] void* getMappedMemory() const { return mapped; }
    [[nodiscard]] VkDeviceAddress getBufferDeviceAddress() const;

private:
    Device& logical_device;
    CommandPool& command_pool;
    VulkanMemoryAllocatorInfo allocator_info;
    VkBuffer buffer;
    uint32_t buffer_size;

    void* mapped{nullptr};
};