#pragma once

#include <memory>

#include "Memory/Buffer.h"

class MemoryAllocator;

class Octree
{
public:
    Octree(MemoryAllocator& memory_allocator);

    const Buffer& aabbBuffer() const { return *aabb_buffer; }

private:
    MemoryAllocator& memory_allocator;

    std::unique_ptr<Buffer> createAABBBuffer();

    std::unique_ptr<Buffer> aabb_buffer;
};
