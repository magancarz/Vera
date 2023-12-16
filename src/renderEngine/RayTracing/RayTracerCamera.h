#pragma once

#include "Ray.h"

class RayTracerCamera
{
public:
    RayTracerCamera(glm::vec3 look_from, glm::vec3 look_at, glm::vec3 vup, float vfov, float aspect);
    __device__ Ray getRay(float u, float v);

private:
    glm::vec3 lower_left_corner{-1.0, -1.0, -1.0};
    glm::vec3 horizontal{1.0, 0.0, 0.0};
    glm::vec3 vertical{0.0, 1.0, 0.0};
    glm::vec3 origin{0.0, 0.0, 0.0};
    glm::vec3 u, v, w;
};
