#pragma once

#include <memory>

#include "Buffer.h"
#include "Image.h"
#include "BufferInfo.h"

class MemoryAllocator
{
public:
    virtual ~MemoryAllocator() = default;

    virtual std::unique_ptr<Buffer> createBuffer(const BufferInfo& buffer_info) = 0;
    virtual std::unique_ptr<Buffer> createStagingBuffer(uint32_t instance_size, uint32_t instance_count) = 0;
    virtual std::unique_ptr<Buffer> createStagingBuffer(uint32_t instance_size, uint32_t instance_count, const void* data) = 0;
    virtual std::unique_ptr<Image> createImage(const VkImageCreateInfo& image_create_info) = 0;
};
