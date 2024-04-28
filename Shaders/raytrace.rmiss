#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "ray.glsl"

layout(location = 0) rayPayloadInEXT Ray payload;

void main()
{
    float t = 0.5 * (payload.direction.y + 1.0);
    vec3 sky_color = (1.0 - t) * vec3(1.0) + t * vec3(0.3, 0.7, 1.0);
    sky_color *= 1.0;
    float light_intensity = payload.depth > 0 ? dot(payload.direction, normalize(vec3(1, 0.5, 1))) : 1.0;
    light_intensity = light_intensity < 0 ? 0 : light_intensity;

    payload.color *= sky_color * light_intensity;
    payload.is_active = 0;
}