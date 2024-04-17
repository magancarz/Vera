#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "ray.glsl"
#include "random.glsl"
#include "cosine_pdf.glsl"
#include "lambertian.glsl"

layout(location = 0) rayPayloadInEXT Ray payload;

struct Vertex
{
    vec3 position;
    uint material_index;
    vec3 normal;
    vec2 uv;
};

struct Material
{
    vec3 color;
    int brightness;
};

layout(binding = 3, set = 0) buffer VertexBuffer { Vertex data[]; } vertex_buffer;
layout(binding = 4, set = 0) buffer IndexBuffer { uint data[]; } index_buffer;

layout(binding = 0, set = 1) buffer MaterialsBuffer { Material data[]; } materials;

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

    ivec3 indices = ivec3(
        index_buffer.data[3 * gl_PrimitiveID + 0],
        index_buffer.data[3 * gl_PrimitiveID + 1],
        index_buffer.data[3 * gl_PrimitiveID + 2]);

    vec3 barycentric = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    Vertex first_vertex = vertex_buffer.data[indices.x];
    Vertex second_vertex = vertex_buffer.data[indices.y];
    Vertex third_vertex = vertex_buffer.data[indices.z];

    vec3 position = first_vertex.position * barycentric.x + second_vertex.position * barycentric.y + third_vertex.position * barycentric.z;
    vec3 geometric_normal = normalize(cross(second_vertex.position - first_vertex.position, third_vertex.position - first_vertex.position));

    Material material = materials.data[first_vertex.material_index];
    vec3 material_color = material.color;

    payload.is_active = material.brightness == 1 ? 0 : 1;
    payload.depth = material.brightness == 1 ? 0 : payload.depth;

    payload.origin = vec4(position, 1.0);

    vec3 u, v, w;
    w = geometric_normal;
    generateOrthonormalBasis(u, v, w);
    payload.direction = vec4(generateRandomDirectionWithCosinePDF(payload.seed, u, v, w), 0.0);

    float cosine_pdf_value = valueFromCosinePDF(payload.direction.xyz, w);
    float scattering_pdf = scatteringPDFFromLambertian(geometric_normal, payload.direction.xyz);
    payload.color *= material_color * scattering_pdf / cosine_pdf_value;

    payload.depth += 1;
}
