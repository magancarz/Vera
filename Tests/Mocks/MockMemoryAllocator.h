#pragma once

#include "Memory/MemoryAllocator.h"

class MockMemoryAllocator : public MemoryAllocator
{
public:
    std::unique_ptr<Buffer> createBuffer(const BufferInfo& buffer_info) override { return nullptr; }
    std::unique_ptr<Buffer> createStagingBuffer(uint32_t instance_size, uint32_t instance_count) override { return nullptr; }
    std::unique_ptr<Buffer> createStagingBuffer(uint32_t instance_size, uint32_t instance_count, const void* data) override { return nullptr; }
    std::unique_ptr<Image> createImage(const VkImageCreateInfo& image_create_info) override { return nullptr; }
};
