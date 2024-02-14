#version 460

layout (location = 0) in vec4 fragment_position;
layout (location = 1) in vec3 fragment_normal;

layout (location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 projection_matrix;
    mat4 view_matrix;
    vec4 ambient_light_color; // w is intensity
    vec3 light_position;
    vec4 light_color;
} ubo;

layout(push_constant) uniform Push
{
    mat4 model_matrix;
    mat4 normal_matrix;
} push;

const float AMBIENT = 0.02;

void main()
{
    vec3 direction_to_light = ubo.light_position - fragment_position.xyz;
    float attenuation = 1.0 / dot(direction_to_light, direction_to_light);

    vec3 light_color = ubo.light_color.xyz * ubo.light_color.w * attenuation;
    vec3 ambient_light = ubo.ambient_light_color.xyz * ubo.ambient_light_color.w;
    vec3 diffuse_light = light_color * max(dot(fragment_normal, normalize(direction_to_light)), 0);
    vec3 light_intensity = (diffuse_light + ambient_light);

    out_color = vec4(light_intensity, 1.0);
}