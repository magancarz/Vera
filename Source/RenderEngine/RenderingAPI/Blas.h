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
            std::unique_ptr<MemoryAllocator>& memory_allocator,
            std::shared_ptr<AssetManager>& asset_manager,
            const MeshComponent* mesh_component);
    ~Blas();

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    Blas(Blas&& other) = default;
    Blas& operator=(Blas&&) = delete;

    BlasInstance createBlasInstance(const glm::mat4& transform);

private:
    VulkanFacade& device;
    std::unique_ptr<MemoryAllocator>& memory_allocator;
    std::shared_ptr<AssetManager> asset_manager;

    void createBlas(const MeshComponent* mesh_component);

    AccelerationStructure blas{};
};
