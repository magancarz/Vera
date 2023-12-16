#pragma once

#include "Ray.h"

class RayTracerCamera
{
public:
    RayTracerCamera(glm::vec3 look_from, glm::vec3 look_at, glm::vec3 camera_up, float fov, float aspect, float aperture = 0, float focus_dist = 1);
    __device__ Ray getRay(curandState* curand_state, float u, float v);
    __device__ Ray getRay(float u, float v);

private:
    glm::vec3 lower_left_corner{-1.0, -1.0, -1.0};
    glm::vec3 horizontal{1.0, 0.0, 0.0};
    glm::vec3 vertical{0.0, 1.0, 0.0};
    glm::vec3 origin{0.0, 0.0, 0.0};
    glm::vec3 u, v, w;

    bool simulate_defocus_blur{false};
    float lens_radius;
};
