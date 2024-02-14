#version 460

layout (location = 0) in vec3 normal_world_space;

layout (location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 projection_view_matrix;
    vec3 direction_to_light;
} ubo;

layout(push_constant) uniform Push
{
    mat4 model_matrix;
    mat4 normal_matrix;
} push;

const float AMBIENT = 0.02;

void main()
{
    float light_intensity = AMBIENT + max(dot(normal_world_space, ubo.direction_to_light), 0);

    out_color = vec4(vec3(light_intensity), 1.0);
}