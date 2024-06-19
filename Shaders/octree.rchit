#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "Commons/ray.glsl"
#include "Commons/random.glsl"
#include "Commons/cosine_pdf.glsl"
#include "Commons/lambertian.glsl"
#include "Commons/material.glsl"
#include "Commons/defines.glsl"
#include "Commons/Common.h"

layout(location = 0) rayPayloadInEXT Ray payload;
layout(location = 1) rayPayloadEXT bool occluded;

layout(binding = 0, set = 0) uniform accelerationStructureEXT top_level_as;

layout(push_constant) uniform _PushConstantRay { PushConstantRay push_constant; };

hitAttributeEXT vec3 attribs;

vec3 randomUnitDirection()
{
    float r1 = rnd(payload.seed);
    float r2 = rnd(payload.seed);
    float x = cos(2 * PI * r1) * 2 * sqrt(r2 * (1 - r2));
    float y = sin(2 * PI * r1) * 2 * sqrt(r2 * (1 - r2));
    float z = 1 - 2 * r2;
    return vec3(x, y, z);
}

vec3 randomToSphere(float radius, float distance_squared)
{
    float r1 = rnd(payload.seed);
    float r2 = rnd(payload.seed);
    float z = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);

    float phi = 2 * PI * r1;
    float x = cos(phi) * sqrt(1 - z * z);
    float y = sin(phi) * sqrt(1 - z * z);

    return vec3(x, y, z);
}

vec3 randomToSun(float radius)
{
    const float distance_squared = 100.0;
    vec3 u, v, w;
    w = push_constant.sun_position;
    generateOrthonormalBasis(u, v, w);
    vec3 random_to_sphere = randomToSphere(radius, distance_squared);
    random_to_sphere = u * random_to_sphere.x + v * random_to_sphere.y + w * random_to_sphere.z;
    return random_to_sphere;
}

void main()
{
    if (payload.is_active == 0)
    {
        return;
    }

    vec3 world_position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.origin = world_position;

    vec3 world_normal = normalize(world_position);
    vec3 absolute_normal = abs(world_normal);
    vec3 sign_normal = sign(world_normal);

    float normal_x = step(absolute_normal.y, absolute_normal.x) * step(absolute_normal.z, absolute_normal.x);
    float normal_y = (1.0 - normal_x) * step(absolute_normal.z, absolute_normal.y);
    float normal_z = 1.0 - normal_x - normal_y;
    world_normal = vec3(normal_x * sign_normal.x, normal_y * sign_normal.y, normal_z * sign_normal.z);

    vec3 positionToLightDirection = randomToSun(push_constant.weather * 10);

    vec3 u, v, w;
    w = world_normal;
    generateOrthonormalBasis(u, v, w);
    vec3 random_cosine_direction = generateRandomDirectionWithCosinePDF(payload.seed, u, v, w);
    payload.direction = rnd(payload.seed) > 0.5 ? positionToLightDirection : random_cosine_direction;

    const uint ray_flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
    occluded = false;
    traceRayEXT(
        top_level_as,
        ray_flags,
        0xFF,
        1,
        0,
        1,
        payload.origin,
        T_MIN,
        payload.direction,
        T_MAX,
        1
    );

    float scattering_pdf = scatteringPDFFromLambertian(payload.direction, world_normal);
    float cosine = occluded ? 0 : max(dot(push_constant.sun_position, payload.direction), 0.0);
    float sun_contribution = cosine * 50;

    payload.color *= sun_contribution * vec3(1, 1, 1) * scattering_pdf;
    payload.depth += 1;
}
