#pragma once

#include "Buffer.h"
#include "BufferInfo.h"

class MemoryAllocator
{
public:
    virtual ~MemoryAllocator() = default;

    virtual std::unique_ptr<Buffer> createBuffer(
            uint32_t instance_size,
            uint32_t instance_count,
            uint32_t usage_flags,
            uint32_t required_memory_flags,
            uint32_t allocation_flags = 0,
            uint32_t preferred_memory_flags = 0,
            uint32_t min_offset_alignment = 0) = 0;
    virtual std::unique_ptr<Buffer> createStagingBuffer(uint32_t instance_size, uint32_t instance_count) = 0;
    virtual std::unique_ptr<Buffer> createStagingBuffer(uint32_t instance_size, uint32_t instance_count, void* data) = 0;
};
