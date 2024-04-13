#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInEXT vec3 hit_value;

struct Vertex
{
    vec3 position;
    vec3 normal;
    vec2 uv;
};

layout(binding = 3, set = 0) buffer VertexBuffer { float data[]; } vertex_buffer;
layout(binding = 4, set = 0) buffer IndexBuffer { uint data[]; } index_buffer;

hitAttributeEXT vec3 attribs;

void main()
{
    ivec3 indices = ivec3(
        index_buffer.data[3 * gl_PrimitiveID + 0],
        index_buffer.data[3 * gl_PrimitiveID + 1],
        index_buffer.data[3 * gl_PrimitiveID + 2]);

    vec3 barycentric = vec3(1.0 - attribs.x - attribs.y,
        attribs.x, attribs.y);

    vec3 vertex_a = vec3(
            vertex_buffer.data[8 * indices.x + 0],
            vertex_buffer.data[8 * indices.x + 1],
            vertex_buffer.data[8 * indices.x + 2]);
    vec3 vertex_b = vec3(
            vertex_buffer.data[8 * indices.y + 0],
            vertex_buffer.data[8 * indices.y + 1],
            vertex_buffer.data[8 * indices.y + 2]);
    vec3 vertex_c = vec3(
            vertex_buffer.data[8 * indices.z + 0],
            vertex_buffer.data[8 * indices.z + 1],
            vertex_buffer.data[8 * indices.z + 2]);

    vec3 position = vertex_a * barycentric.x + vertex_b * barycentric.y +
                  vertex_c * barycentric.z;

    vec3 geometric_normal = normalize(cross(vertex_b - vertex_a, vertex_c - vertex_a));

    vec3 light_direction = vec3(0.0, 1.0, 0.0);
    float light_intensity = dot(geometric_normal, light_direction);

    hit_value = vec3(1.0, 1.0, 1.0) * light_intensity;
}
