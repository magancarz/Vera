#pragma once

#include <memory>

#include "RenderEngine/Models/AccelerationStructure.h"
#include "RenderEngine/Models/Model.h"
#include "Objects/Components/MeshComponent.h"
#include "RenderEngine/Memory/MemoryAllocator.h"

class Blas
{
public:
    Blas(VulkanFacade& device, std::unique_ptr<MemoryAllocator>& memory_allocator, const MeshComponent* mesh_component);
    ~Blas();

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    Blas(Blas&& other) = default;
    Blas& operator=(Blas&&) = delete;

    BlasInstance createBlasInstance(const glm::mat4& transform);

private:
    VulkanFacade& device;
    std::unique_ptr<MemoryAllocator>& memory_allocator;

    void createBlas(const MeshComponent* mesh_component);

    AccelerationStructure blas{};
};
