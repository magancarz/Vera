#version 330 core

in vec2 pass_texture_coords;

out vec4 out_Color;

uniform sampler2D g_position;
uniform sampler2D g_normal;
uniform sampler2D g_color_spec;
uniform vec3 view_position;

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

void main(void)
{
    vec3 fragment_position = texture(g_position, pass_texture_coords).rgb;
    vec3 normal = texture(g_normal, pass_texture_coords).rgb;
    vec3 color = texture(g_color_spec, pass_texture_coords).rgb;
    float reflectivity = texture(g_color_spec, pass_texture_coords).a;

    vec3 lighting = color * 0.1;
    vec3 view_direction = normalize(view_position - fragment_position);
    vec3 light_direction = normalize(light.light_position - fragment_position);
    vec3 diffuse = max(dot(normal, light_direction), 0.0) * color * light.light_color;
    lighting += diffuse;
    out_Color = vec4(lighting, 1.0);
}