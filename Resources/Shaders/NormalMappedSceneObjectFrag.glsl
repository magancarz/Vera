#version 330 core

in vec4 fragment_world_position;
in vec2 pass_texture_coords;
in mat3 TBN;

layout (location = 0) out vec3 g_position;
layout (location = 1) out vec3 g_normal;
layout (location = 2) out vec4 g_color_spec;

uniform sampler2D color_texture_sampler;
uniform sampler2D normal_texture_sampler;

uniform float reflectivity;

void main(void)
{
	vec3 sampled_normal = texture(normal_texture_sampler, pass_texture_coords).rgb;
	sampled_normal = normalize(sampled_normal * 2.0 - 1.0);
	sampled_normal = TBN * sampled_normal;

	g_position = fragment_world_position.xyz;
	g_normal = sampled_normal;
	g_color_spec.rgb = texture(color_texture_sampler, pass_texture_coords).rgb;
	g_color_spec.a = reflectivity;
}