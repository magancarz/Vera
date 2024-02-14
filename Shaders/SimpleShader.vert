#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec2 uv;

layout (location = 0) out vec4 fragment_position;
layout (location = 1) out vec3 fragment_normal;

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 projection_view_matrix;
    vec4 ambient_light_color; // w is intensity
    vec3 light_position;
    vec4 light_color;
} ubo;

layout(push_constant) uniform Push
{
    mat4 model_matrix;
    mat4 normal_matrix;
} push;

void main()
{
    fragment_position = push.model_matrix * vec4(position, 1.0);
    gl_Position = ubo.projection_view_matrix * fragment_position;
    fragment_normal = normalize(mat3(push.normal_matrix) * normal);
}