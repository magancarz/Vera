#version 460

layout (location = 0) in vec4 fragment_position;
layout (location = 1) in vec3 fragment_normal;
layout (location = 2) in vec2 fragment_uv;

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
    mat4 inverse_view_matrix;
    vec4 ambient_light_color; // w is intensity
    PointLight point_lights[10];
    int number_of_lights;
} ubo;

layout(set = 1, binding = 0) uniform sampler2D image;

layout(push_constant) uniform Push
{
    mat4 model_matrix;
    mat4 normal_matrix;
} push;

const float AMBIENT = 0.02;

void main()
{
    vec3 ambient_light = ubo.ambient_light_color.xyz * ubo.ambient_light_color.w;
    vec3 specular_light = vec3(0.0);
    vec3 normal = normalize(fragment_normal);

    vec3 camera_position_world = ubo.inverse_view_matrix[3].xyz;
    vec3 view_direction = normalize(camera_position_world - fragment_position.xyz);

    vec3 diffuse_light = ambient_light;
    for (int i = 0; i < ubo.number_of_lights; ++i)
    {
        PointLight light = ubo.point_lights[i];
        vec3 direction_to_light = light.position.xyz - fragment_position.xyz;
        float attenuation = 1.0 / dot(direction_to_light, direction_to_light);
        direction_to_light = normalize(direction_to_light);

        float cos_angle_incidence = max(dot(normal, direction_to_light), 0);
        vec3 intensity = light.color.xyz * light.color.w * attenuation;
        diffuse_light += intensity * cos_angle_incidence;

        vec3 half_angle = normalize(direction_to_light + view_direction);
        float blinn_term = dot(normal, half_angle);
        blinn_term = clamp(blinn_term, 0, 1);
        blinn_term = pow(blinn_term, 512.0);
        specular_light += intensity * blinn_term;
    }

//    vec3 texture_color = texture(image, fragment_uv).rgb;
    vec3 texture_color = vec3(1.0);
    out_color = vec4((diffuse_light + specular_light) * texture_color, 1.0);
}