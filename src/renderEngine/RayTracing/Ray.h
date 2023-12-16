#pragma once

#include <cuda/std/chrono>

#include <corecrt_math_defines.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>

class Ray
{
public:
    Ray() = default;

    __device__ Ray(const glm::vec3& origin, const glm::vec3& b)
        : origin{ origin }, direction{ b }, inv_dir{ 1.f / direction.x, 1.f / direction.y, 1.f / direction.z },
        is_dir_neg{ direction.x < 0, direction.y < 0, direction.z < 0} {}

    __device__ glm::vec3 pointAtParameter(const float t) const
    {
        return origin + t * direction;
    }

    glm::vec3 origin;
    glm::vec3 direction;
    float min = 0.0001f;
    float max = FLT_MAX;
    glm::vec3 inv_dir;
    bool is_dir_neg[3];
    curandState* curand_state;
};
