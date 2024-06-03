#pragma once

#include <memory>

#include <glm/glm.hpp>

#include "RenderEngine/Models/AccelerationStructure.h"
#include "RenderEngine/Models/BlasInstance.h"

class MeshComponent;
class MemoryAllocator;
class AssetManager;

class Blas
{
public:
    Blas(
            VulkanFacade& device,
            MemoryAllocator& memory_allocator,
            AssetManager& asset_manager,
            MeshComponent& mesh_component);
    ~Blas();

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    Blas(Blas&& other) = default;
    Blas& operator=(Blas&&) = delete;

    BlasInstance createBlasInstance(const glm::mat4& transform);

private:
    VulkanFacade& device;
    MemoryAllocator& memory_allocator;
    AssetManager& asset_manager;

    void createBlas(MeshComponent& mesh_component);

    AccelerationStructure blas{};
};
