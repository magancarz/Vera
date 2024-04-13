#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInEXT vec3 hit_value;

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
};

layout(binding = 3, set = 0) buffer VertexBuffer { Vertex data[]; } vertex_buffer;
layout(binding = 4, set = 0) buffer IndexBuffer { uint data[]; } index_buffer;

layout(binding = 0, set = 1) buffer MaterialsBuffer { Material data[]; } materials;

hitAttributeEXT vec3 attribs;

void main()
{
    ivec3 indices = ivec3(
        index_buffer.data[3 * gl_PrimitiveID + 0],
        index_buffer.data[3 * gl_PrimitiveID + 1],
        index_buffer.data[3 * gl_PrimitiveID + 2]);

    vec3 barycentric = vec3(1.0 - attribs.x - attribs.y,
        attribs.x, attribs.y);

    Vertex first_vertex = vertex_buffer.data[indices.x];
    Vertex second_vertex = vertex_buffer.data[indices.y];
    Vertex third_vertex = vertex_buffer.data[indices.z];

    vec3 geometric_normal = normalize(cross(second_vertex.position - first_vertex.position, third_vertex.position - first_vertex.position));

    vec3 light_direction = vec3(0.0, 1.0, 0.0);
    float light_intensity = dot(geometric_normal, light_direction);

    vec3 material_color = materials.data[first_vertex.material_index].color;

    hit_value = material_color * light_intensity;
}
