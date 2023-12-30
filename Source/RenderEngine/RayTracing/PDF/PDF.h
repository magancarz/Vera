#pragma once

#include <curand_kernel.h>

#include "glm/vec3.hpp"

class PDF
{
public:
    __device__ PDF(curandState* curand_state);
    __device__ virtual ~PDF() {}

    __device__ virtual float value(const glm::vec3& direction) const = 0;
    __device__ virtual glm::vec3 generate() const = 0;

protected:
    curandState* curand_state;
};
