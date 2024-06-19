#pragma once

#include "Blas.h"
#include "RenderEngine/AccelerationStructures/Octree/Octree.h"

class OctreeBlas : public Blas
{
public:
    OctreeBlas(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        const Octree& octree);

    OctreeBlas(const OctreeBlas&) = delete;
    OctreeBlas& operator=(const OctreeBlas&) = delete;

    void createBlas() override;

protected:
    const Octree& octree;
};
