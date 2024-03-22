#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec4 fragment_position;
layout (location = 1) out vec3 fragment_normal;
layout (location = 2) out vec2 fragment_uv;

struct PointLight
{
    vec4 position;
    vec4 color;
};

layout(set = 0, binding = 0) uniform GlobalUbo
{
    mat4 projection_matrix;
    mat4 view_matrix;
    mat4 inverse_view_matrix;
    vec4 ambient_light_color; // w is intensity
    PointLight point_lights[10];
    int number_of_lights;
} ubo;

layout(push_constant) uniform Push
{
    mat4 model_matrix;
    mat4 normal_matrix;
} push;

void main()
{
    fragment_uv = uv;
    fragment_position = push.model_matrix * vec4(position, 1.0);
    gl_Position = ubo.projection_matrix * ubo.view_matrix * fragment_position;
    fragment_normal = normalize(mat3(push.normal_matrix) * normal);
}