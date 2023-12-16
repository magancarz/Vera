#include "RayTracerCamera.h"

RayTracerCamera::RayTracerCamera(glm::vec3 look_from, glm::vec3 look_at, glm::vec3 vup, float vfov, float aspect)
{
    const float theta = vfov * 3.14159f / 180.0f;
    const float half_height = tan(theta / 2);
    const float half_width = aspect * half_height;

    origin = look_from;
    w = normalize(look_from - look_at);
    u = normalize(cross(vup, w));
    v = cross(w, u);

    lower_left_corner = origin - half_width * u - half_height * v - w;
    horizontal = 2 * half_width * u;
    vertical = 2 * half_height * v;
}

__device__ Ray RayTracerCamera::getRay(float u, float v)
{
    return {origin, lower_left_corner + u * horizontal + v * vertical - origin};
}
