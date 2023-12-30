#pragma once

#include <glm/glm.hpp>

struct OrthonormalBasis
{
    glm::vec3 x;
    glm::vec3 y;
    glm::vec3 z;

    __device__ glm::vec3 local(float a, float b, float c)
    {
        return a * x + b * y + c * z;
    }

    __device__ glm::vec3 local(const glm::vec3& a) const
    {
        return a.x * x + a.y * y + a.z * z;
    }

    __device__ void buildFromVector(const glm::vec3& a)
    {
        z = normalize(a);
        const glm::vec3 temp = fabs(z.x) > 0.9 ? glm::vec3{0, 1, 0} : glm::vec3{1, 0, 0};
        y = normalize(cross(z, temp));
        x = cross(z, y);
    }
};
