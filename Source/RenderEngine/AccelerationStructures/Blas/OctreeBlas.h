#pragma once

#include "Blas.h"
#include "RenderEngine/AccelerationStructures/Octree/Octree.h"

class OctreeBlas : public Blas
{
public:
    OctreeBlas(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        Octree octree);

    OctreeBlas(const OctreeBlas&) = delete;
    OctreeBlas& operator=(const OctreeBlas&) = delete;

    void createBlas() override;

    [[nodiscard]] const Buffer& octreeBuffer() const
    {
        assert(octree_buffer && "Called octreeBuffer on OctreeBlas before creating blas!");
        return *octree_buffer;
    }

protected:
    Octree octree;

    void allocateOctreeNodesOnGPUMemory();

    std::unique_ptr<Buffer> octree_buffer;

    void allocateOctreeAABBOnGPUMemory();

    std::unique_ptr<Buffer> aabb_buffer;
};
