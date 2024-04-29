#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "ray.glsl"

layout(location = 0) rayPayloadInEXT Ray payload;

void main()
{
    const vec3 SUN = normalize(vec3(0, 1, 1));
    float sun_amount = max(dot(payload.direction, SUN), 0.0);
    vec3 sun_color = vec3(0.6, 0.4, 0.2);

    vec3  sky = mix(vec3(.75, .73, .71), vec3(.5, .7, .9) , 0.25 + payload.direction.z);
    sky = sky + sun_color * min(pow(sun_amount, 1500.0) * 5.0, 1.0);
    sky = sky + sun_color * min(pow(sun_amount, 10.0) * .6, 1.0);

    vec3 sky_color = clamp(sky + vec3(SUN.z - 1.0) * vec3(0.5 , .75, 1.0), 0.0, 10.0);
    payload.color *= sky_color;
    payload.is_active = 0;
}