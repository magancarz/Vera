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

layout(location = 0) rayPayloadInEXT Ray payload;

struct Material
{
    vec3 color;
    int brightness;
};

struct ObjectDescription
{
    uint64_t vertex_address;
    uint64_t index_address;
    uint64_t material_address;
};
layout(binding = 3, set = 0) buffer ObjectDescriptions { ObjectDescription data[]; } object_descriptions;

layout(binding = 0, set = 1) buffer MaterialsBuffer { Material data[]; } materials;

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

hitAttributeEXT vec3 attribs;

vec3 randomInHemisphere(uint seed, vec3 normal)
{
    float x = rnd(seed) * 2.0 - 1.0;
    float y = rnd(seed) * 2.0 - 1.0;
    float z = rnd(seed) * 2.0 - 1.0;

    vec3 random = vec3(x, y, z);
    random = normalize(random);
    random *= sign(dot(random, normal));
    return random;
}

void main()
{
    if (payload.is_active == 0)
    {
        return;
    }

    ObjectDescription object_description = object_descriptions.data[gl_InstanceCustomIndexEXT];

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
    geometric_normal = normalize(vec3(geometric_normal * gl_WorldToObjectEXT));

    vec3 position = first_vertex.position * barycentric.x + second_vertex.position * barycentric.y + third_vertex.position * barycentric.z;
    position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

    MaterialBuffer material_buffer = MaterialBuffer(object_description.material_address);
    Material material = material_buffer.m;
    vec3 material_color = material.color;

    payload.is_active = material.brightness == 1 ? 0 : 1;

    payload.origin = vec4(position, 1.0);

    vec3 u, v, w;
    w = geometric_normal;
    generateOrthonormalBasis(u, v, w);
    payload.direction = vec4(generateRandomDirectionWithCosinePDF(payload.seed, u, v, w), 0.0);

    float cosine_pdf_value = valueFromCosinePDF(payload.direction.xyz, w);
    float scattering_pdf = scatteringPDFFromLambertian(geometric_normal, payload.direction.xyz);
    payload.color *= material_color * scattering_pdf / cosine_pdf_value;
}
