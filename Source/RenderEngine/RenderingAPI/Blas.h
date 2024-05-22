#pragma once

#include <memory>

#include "RenderEngine/Models/AccelerationStructure.h"
#include "RenderEngine/Models/Model.h"

class Blas
{
public:
    Blas(VulkanFacade& device, const std::shared_ptr<Model>& model);
    ~Blas();

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    Blas(Blas&& other) = default;
    Blas& operator=(Blas&&) = delete;

    BlasInstance createBlasInstance(const glm::mat4& transform, uint32_t id);

private:
    VulkanFacade& device;

    void createBlas(const std::shared_ptr<Model>& model);

    AccelerationStructure blas{};
};
