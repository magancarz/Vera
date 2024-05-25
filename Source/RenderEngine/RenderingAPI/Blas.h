#pragma once

#include <memory>

#include "RenderEngine/Models/AccelerationStructure.h"
#include "RenderEngine/Models/Model.h"
#include "Objects/Components/MeshComponent.h"

class Blas
{
public:
    Blas(VulkanFacade& device, const MeshComponent* mesh_component);
    ~Blas();

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    Blas(Blas&& other) = default;
    Blas& operator=(Blas&&) = delete;

    BlasInstance createBlasInstance(const glm::mat4& transform);

private:
    VulkanFacade& device;

    void createBlas(const MeshComponent* mesh_component);

    AccelerationStructure blas{};
};
