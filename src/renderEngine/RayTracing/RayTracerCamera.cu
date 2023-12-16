#include "RayTracerCamera.h"

#include "utils/CurandUtils.h"

RayTracerCamera::RayTracerCamera(glm::vec3 look_from, glm::vec3 look_at, glm::vec3 camera_up, float fov, float aspect, float aperture, float focus_dist)
{
    simulate_defocus_blur = aperture > 0;
    lens_radius = aperture / 2.f;
    const float theta = fov * static_cast<float>(M_PI) / 180.0f;
    const float half_height = tan(theta / 2);
    const float half_width = aspect * half_height;

    origin = look_from;
    w = normalize(look_from - look_at);
    u = normalize(cross(camera_up, w));
    v = cross(w, u);

    lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
    horizontal = 2.f * half_width * focus_dist * u;
    vertical = 2.f * half_height * focus_dist * v;
}

__device__ Ray RayTracerCamera::getRay(curandState* curand_state, float u, float v)
{
    glm::vec3 offset{0};
    if (simulate_defocus_blur)
    {
        const glm::vec3 random_in_disk = lens_radius * randomInUnitDisk(curand_state);
        offset = glm::vec3{this->u * random_in_disk.x + this->v * random_in_disk.y};
    }
    return {origin + offset, lower_left_corner + u * horizontal + v * vertical - origin - offset};
}

__device__ Ray RayTracerCamera::getRay(float u, float v)
{
    return {origin, lower_left_corner + u * horizontal + v * vertical - origin};
}
