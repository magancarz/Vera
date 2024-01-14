#version 330 core

in vec2 pass_texture_coords;

out vec4 out_Color;

uniform sampler2D g_position;
uniform sampler2D g_normal;
uniform sampler2D g_color_spec;
uniform sampler2D ssao;

uniform samplerCube shadow_map;

struct Light
{
    vec3 light_position;
    vec3 light_color;
    vec3 attenuation;
};
layout (std140) uniform LightInfos
{
    Light light;
};

layout (std140) uniform TransformationMatrices
{
    mat4 model;
    mat4 view;
    mat4 proj;
};

uniform vec3 view_position;

const float ambient = 0.1;
vec3 sample_offset_directions[20] = vec3[]
(
vec3( 1, 1, 1), vec3( 1, -1, 1), vec3(-1, -1, 1), vec3(-1, 1, 1),
vec3( 1, 1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1),
vec3( 1, 1, 0), vec3( 1, -1, 0), vec3(-1, -1, 0), vec3(-1, 1, 0),
vec3( 1, 0, 1), vec3(-1, 0, 1), vec3( 1, 0, -1), vec3(-1, 0, -1),
vec3( 0, 1, 1), vec3( 0, -1, 1), vec3( 0, -1, -1), vec3( 0, 1, -1)
);

float calculateSpecularValue(vec3 position, vec3 normal, float reflectivity, vec3 to_light_dir, float attenuation);
float calculateBrightnessFromPointLight(vec3 normal, vec3 to_light_dir, float attenuation);
float pointLightShadowCalculation(vec3 position);

void main(void)
{
    vec3 position = texture(g_position, pass_texture_coords).rgb;
    vec3 normal = texture(g_normal, pass_texture_coords).rgb;
    vec3 color = texture(g_color_spec, pass_texture_coords).rgb;
    float reflectivity = texture(g_color_spec, pass_texture_coords).a;
    float ambient_occlusion = texture(ssao, pass_texture_coords).r;

    vec3 world_space_position = vec3(inverse(view) * vec4(position, 1.0));
    vec3 world_space_normal = mat3(inverse(view)) * normal;

    vec3 to_light_vector = light.light_position - world_space_position;
    float distance = length(to_light_vector);
    float attenuation = 1.0f / (light.attenuation.x + light.attenuation.y * distance + light.attenuation.z * (distance * distance));
    float brightness = calculateBrightnessFromPointLight(world_space_normal, to_light_vector, attenuation);
    float specular = calculateSpecularValue(world_space_position, world_space_normal, reflectivity, to_light_vector, attenuation);
    float shadow = pointLightShadowCalculation(world_space_position);

    vec3 total_diffuse = brightness * light.light_color;
    vec3 total_specular = specular * light.light_color;
    total_diffuse = max(total_diffuse, 0);
    total_specular = max(total_specular, 0);

    vec3 lighting = (ambient + (1.0 - shadow) * (total_diffuse + total_specular));
    vec3 final_color = lighting * color * ambient_occlusion;
    out_Color = vec4(final_color, 1.0);
}

float calculateSpecularValue(vec3 position, vec3 normal, float reflectivity, vec3 to_light_dir, float attenuation)
{
    vec3 normalized_to_light_dir = normalize(to_light_dir);
    vec3 to_camera_vector = normalize(view_position - position);
    vec3 halfway_dir = normalize(normalized_to_light_dir + to_camera_vector);
    float specular = pow(max(dot(normal, halfway_dir), 0.0), 32.0) / attenuation * reflectivity;

    return specular;
}

float calculateBrightnessFromPointLight(vec3 normal, vec3 to_light_dir, float attenuation)
{
    vec3 normalized_to_light_dir = normalize(to_light_dir);
    float n_dot1 = dot(normal, normalized_to_light_dir);
    float brightness = max(n_dot1 * attenuation, 0.0);

    return brightness;
}

float pointLightShadowCalculation(vec3 position)
{
    vec3 fragment_to_light = position - light.light_position;
    float current_depth = length(fragment_to_light);

    float shadow = 0.0;
    float bias = -0.00005;
    float samples = 20;
    float offset = 0.1;
    float view_distance = length(position);
    float disk_radius = (1.0 + (view_distance / 30.0)) / 25.0;

    for(int i = 0; i < samples; ++i)
    {
        float closest_depth = texture(shadow_map, fragment_to_light + sample_offset_directions[i] * disk_radius).r;
        closest_depth *= 30.0;
        if(current_depth - bias > closest_depth)
            shadow += 1.0;
    }
    shadow /= float(samples);

    return shadow;
}