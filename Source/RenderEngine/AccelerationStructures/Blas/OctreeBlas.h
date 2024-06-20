#pragma once

#include "Blas.h"
#include "RenderEngine/AccelerationStructures/Octree/Octree.h"

class OctreeBlas : public Blas
{
public:
    OctreeBlas(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        Octree  octree);

    OctreeBlas(const OctreeBlas&) = delete;
    OctreeBlas& operator=(const OctreeBlas&) = delete;

    void createBlas() override;

protected:
    Octree octree;
    std::unique_ptr<Buffer> octree_buffer;
};
