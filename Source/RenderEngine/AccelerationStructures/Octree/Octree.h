#pragma once

#include <memory>

#include "Memory/Buffer.h"
#include "Voxel.h"

class MemoryAllocator;

class Octree
{
public:
    explicit Octree(MemoryAllocator& memory_allocator);

    [[nodiscard]] const Buffer& aabbBuffer() const { return *aabb_buffer; }

private:
    MemoryAllocator& memory_allocator;

    std::unique_ptr<Buffer> createAABBBuffer();

    std::unique_ptr<Buffer> aabb_buffer;
};
