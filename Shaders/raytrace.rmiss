#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "ray.glsl"

layout(location = 0) rayPayloadInEXT Ray payload;

void main()
{
    float t = 0.5 * (payload.direction.y + 1.0);
    vec3 sky_color = (1.0 - t) * vec3(1.0) + t * vec3(0.5, 0.7, 1.0);
    payload.color *= sky_color;
    payload.is_active = 0;
}