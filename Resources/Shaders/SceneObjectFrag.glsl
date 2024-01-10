#version 330 core

layout (location = 0) out vec3 g_position;
layout (location = 1) out vec3 g_normal;
layout (location = 2) out vec4 g_color_spec;

in vec4 fragment_world_position;
in vec2 pass_texture_coords;
in vec3 surface_normal;

uniform sampler2D color_texture_sampler;

uniform float reflectivity;

void main(void)
{
	g_position = fragment_world_position.xyz;
	g_normal = surface_normal;
	g_color_spec.rgb = texture(color_texture_sampler, pass_texture_coords).rgb;
	g_color_spec.a = reflectivity;
}