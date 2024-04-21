#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "ray.glsl"
#include "random.glsl"
#include "cosine_pdf.glsl"
#include "lambertian.glsl"
#include "material.glsl"
#include "defines.glsl"

layout(location = 0) rayPayloadInEXT Ray payload;
layout(location = 1) rayPayloadEXT bool is_shadow;

struct ObjectDescription
{
    uint64_t vertex_address;
    uint64_t index_address;
    uint64_t material_address;
    uint64_t num_of_triangles;
    mat4 object_to_world;
    float surface_area;
};

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 3, set = 0) buffer ObjectDescriptions { ObjectDescription data[]; } object_descriptions;
layout(binding = 4, set = 0) buffer LightIndices { uint l[]; } light_indices;

struct Vertex
{
    vec3 position;
    uint alignment1;
    vec3 normal;
    uint alignment2;
    vec2 uv;
    vec2 alignment3;
};

layout(buffer_reference, scalar) readonly buffer Vertices { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uint i[]; };
layout(buffer_reference, scalar) readonly buffer MaterialBuffer { Material m; };

layout(push_constant) uniform PushConstantRay
{
    uint time;
    uint frames;
    uint number_of_lights;
} push_constant;

hitAttributeEXT vec3 attribs;

void main()
{
    if (payload.is_active == 0)
    {
        return;
    }

    ObjectDescription object_description = object_descriptions.data[gl_InstanceCustomIndexEXT];

    MaterialBuffer material_buffer = MaterialBuffer(object_description.material_address);
    Material material = material_buffer.m;
    vec3 material_color = material.color;

    Indices index_buffer = Indices(object_description.index_address);
    uvec3 indices = uvec3(index_buffer.i[3 * gl_PrimitiveID + 0],
                          index_buffer.i[3 * gl_PrimitiveID + 1],
                          index_buffer.i[3 * gl_PrimitiveID + 2]);

    Vertices vertex_buffer = Vertices(object_description.vertex_address);
    Vertex first_vertex = vertex_buffer.v[indices.x];
    Vertex second_vertex = vertex_buffer.v[indices.y];
    Vertex third_vertex = vertex_buffer.v[indices.z];

    vec3 barycentric = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 geometric_normal = first_vertex.normal * barycentric.x + second_vertex.normal * barycentric.y + third_vertex.normal * barycentric.z;
    geometric_normal = normalize(geometric_normal);

    vec3 position = first_vertex.position * barycentric.x + second_vertex.position * barycentric.y + third_vertex.position * barycentric.z;
    position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

    ObjectDescription random_light = object_descriptions.data[light_indices.l[uint(floor(rnd(payload.seed) * float(push_constant.number_of_lights)))]];
    Indices light_index_buffer = Indices(random_light.index_address);

    uint light_random_primitive = uint(floor(rnd(payload.seed) * float(random_light.num_of_triangles)));

    uvec3 light_indices = uvec3(light_index_buffer.i[3 * light_random_primitive + 0],
                                light_index_buffer.i[3 * light_random_primitive + 1],
                                light_index_buffer.i[3 * light_random_primitive + 2]);

    Vertices light_vertices = Vertices(random_light.vertex_address);
    Vertex light_first_vertex = light_vertices.v[light_indices.x];
    Vertex light_second_vertex = light_vertices.v[light_indices.y];
    Vertex light_third_vertex = light_vertices.v[light_indices.z];

    vec2 uv = vec2(rnd(payload.seed), rnd(payload.seed));
    if (uv.x + uv.y > 1.0f)
    {
        uv.x = 1.0f - uv.x;
        uv.y = 1.0f - uv.y;
    }

    vec3 light_barycentric = vec3(1.0 - uv.x - uv.y, uv.x, uv.y);
    vec3 light_normal = light_first_vertex.normal * light_barycentric.x + light_second_vertex.normal * light_barycentric.y + light_third_vertex.normal * light_barycentric.z;
    light_normal = normalize(light_normal);

    vec3 light_position = light_first_vertex.position * light_barycentric.x + light_second_vertex.position * light_barycentric.y + light_third_vertex.position * light_barycentric.z;
    light_position = vec3(random_light.object_to_world * vec4(light_position, 1.0));

    vec3 positionToLightDirection = normalize(light_position - position);
    vec3 shadowRayDirection = positionToLightDirection;
    float shadowRayDistance = length(light_position - position) - T_MIN;

    payload.origin = vec4(position, 1.0);

    vec3 u, v, w;
    w = geometric_normal;
    generateOrthonormalBasis(u, v, w);
    vec4 random_cosine_direction = vec4(generateRandomDirectionWithCosinePDF(payload.seed, u, v, w), 0.0);
    payload.direction = rnd(payload.seed) > 0.5 ? random_cosine_direction : vec4(shadowRayDirection, 0.0);

    float cosine = abs(dot(payload.direction.xyz, light_normal));
    cosine = cosine < 0.00000001f ? 0.00000001f : cosine;
    float distance_squared = shadowRayDistance * shadowRayDistance;
    float sampling_pdf_value = distance_squared / (cosine * random_light.surface_area);

    float cosine_pdf_value = valueFromCosinePDF(payload.direction.xyz, w);

    float final_pdf_value = (sampling_pdf_value + cosine_pdf_value) * 0.5f;
    float scattering_pdf = scatteringPDFFromLambertian(geometric_normal, payload.direction.xyz);
    payload.color *= material_color * scattering_pdf / final_pdf_value;
}
