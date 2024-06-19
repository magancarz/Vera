#pragma once

#include <Assets/Mesh.h>
#include <glm/glm.hpp>

#include "RenderEngine/AccelerationStructures/AccelerationStructure.h"
#include "RenderEngine/AccelerationStructures/BlasBuilder.h"
#include "RenderEngine/AccelerationStructures/BlasInstance.h"

class MeshComponent;
class MemoryAllocator;
class AssetManager;

class Blas
{
public:
    Blas(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator);
    virtual ~Blas() = default;

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;

    virtual void createBlas() = 0;
    [[nodiscard]] BlasInstance createBlasInstance(const glm::mat4& transform) const;

    void update();

protected:
    VulkanHandler& device;
    MemoryAllocator& memory_allocator;

    BlasBuilder::BlasInput blas_input{};
    AccelerationStructure blas{device.getLogicalDevice()};
};
