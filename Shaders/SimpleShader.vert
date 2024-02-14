#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec2 uv;

layout (location = 0) out vec3 normal_world_space;

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 projection_view_matrix;
    vec3 direction_to_light;
} ubo;

layout(push_constant) uniform Push
{
    mat4 model_matrix;
    mat4 normal_matrix;
} push;

void main()
{
    gl_Position = ubo.projection_view_matrix * push.model_matrix * vec4(position, 1.0);
    normal_world_space = normalize(mat3(push.normal_matrix) * normal);
}