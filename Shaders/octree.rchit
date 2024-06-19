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

layout(binding = 0, set = 0) uniform accelerationStructureEXT top_level_as;

layout(push_constant) uniform _PushConstantRay { PushConstantRay push_constant; };

hitAttributeEXT vec3 attribs;

void main()
{
    if (payload.is_active == 0)
    {
        return;
    }

    payload.color = vec3(1, 1, 0);
    payload.is_active = 0;
}
