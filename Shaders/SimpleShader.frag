#version 460

layout (location = 0) in vec4 fragment_position;
layout (location = 1) in vec3 fragment_normal;

layout (location = 0) out vec4 out_color;

struct PointLight
{
    vec4 position;
    vec4 color;
};

layout(set = 0, binding = 0) uniform GlobalUbo
{
    mat4 projection_matrix;
    mat4 view_matrix;
    vec4 ambient_light_color; // w is intensity
    PointLight point_lights[10];
    int number_of_lights;
} ubo;

layout(push_constant) uniform Push
{
    mat4 model_matrix;
    mat4 normal_matrix;
} push;

const float AMBIENT = 0.02;

void main()
{
    vec3 ambient_light = ubo.ambient_light_color.xyz * ubo.ambient_light_color.w;
    vec3 normal = normalize(fragment_normal);

    vec3 diffuse_light = ambient_light;
    for (int i = 0; i < ubo.number_of_lights; ++i)
    {
        PointLight light = ubo.point_lights[i];
        vec3 direction_to_light = light.position.xyz - fragment_position.xyz;
        float attenuation = 1.0 / dot(direction_to_light, direction_to_light);
        float cos_angle_incidence = max(dot(normal, normalize(direction_to_light)), 0);
        vec3 intensity = light.color.xyz * light.color.w * attenuation;
        diffuse_light += intensity * cos_angle_incidence;
    }

    out_color = vec4(diffuse_light, 1.0);
}