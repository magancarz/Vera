#pragma once

#include <Assets/Mesh.h>
#include <glm/glm.hpp>

#include "AccelerationStructure.h"
#include "BlasBuilder.h"
#include "BlasInstance.h"

class MeshComponent;
class MemoryAllocator;
class AssetManager;

class Blas
{
public:
    Blas(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        AssetManager& asset_manager,
        const Mesh& mesh);
    ~Blas();

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    Blas(Blas&& other) = default;
    Blas& operator=(Blas&&) = delete;

    [[nodiscard]] BlasInstance createBlasInstance(const glm::mat4& transform) const;

    void update();

private:
    VulkanHandler& device;
    MemoryAllocator& memory_allocator;
    AssetManager& asset_manager;

    void createBlas(const Mesh& mesh);

    BlasBuilder::BlasInput blas_input{};
    AccelerationStructure blas{};
};
