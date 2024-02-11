#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec2 uv;

layout (location = 0) out vec3 normal_world_space;

layout(push_constant) uniform Push
{
    mat4 transform;
    mat4 normal_matrix;
} push;

void main()
{
    gl_Position = push.transform * vec4(position, 1.0);
    normal_world_space = normalize(mat3(push.normal_matrix) * normal);
}