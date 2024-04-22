#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "ray.glsl"
#include "random.glsl"
#include "ray_random.glsl"
#include "lambertian.glsl"
#include "material.glsl"
#include "defines.glsl"

layout(location = 0) rayPayloadInEXT Ray payload;

struct ObjectDescription
{
    uint64_t vertex_address;
    uint64_t index_address;
    uint64_t material_address;
    uint64_t num_of_triangles;
    mat4 object_to_world;
    float surface_area;
};

layout(binding = 3, set = 0) buffer ObjectDescriptions { ObjectDescription data[]; } object_descriptions;

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

    payload.origin = position;

    const float NdotD = dot(geometric_normal, payload.direction.xyz);
    vec3 refrNormal = geometric_normal;
    float refrEta;

    if(NdotD > 0.0f) {
        refrNormal = -geometric_normal;
        refrEta = 1.0f / material.refractive_index;
    } else {
        refrNormal = geometric_normal;
        refrEta = material.refractive_index;
    }

    vec3 refraction = refract(payload.direction.xyz, refrNormal, refrEta);
    vec3 reflection = reflect(payload.direction.xyz, refrNormal);

    float cos_theta = dot(-payload.direction.xyz, geometric_normal);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refrEta * sin_theta > 1.0;
    vec3 final_direction = cannot_refract ? reflection : refraction;
    payload.direction = final_direction;
    payload.color *= material_color;
}
