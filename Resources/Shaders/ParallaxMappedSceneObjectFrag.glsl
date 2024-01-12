#version 330 core

in vec4 fragment_world_position;
in vec4 tangent_space_fragment_world_position;
in vec2 pass_texture_coords;
in vec3 tangent_space_view_position;
in mat3 TBN;

layout (location = 0) out vec3 g_position;
layout (location = 1) out vec3 g_normal;
layout (location = 2) out vec4 g_color_spec;

uniform sampler2D color_texture_sampler;
uniform sampler2D normal_texture_sampler;
uniform sampler2D depth_texture_sampler;

uniform float reflectivity;
uniform float height_scale;

vec2 parallaxMapping(vec2 tex_coords, vec3 view_dir)
{
	const float min_num_of_layers = 10;
	const float max_num_of_layers = 32;
	float num_of_layers = mix(max_num_of_layers, min_num_of_layers, abs(dot(vec3(0.0, 0.0, 1.0), view_dir)));

	float layer_depth = 1.0 / num_of_layers;
	float current_layer_depth = 0.0;
	vec2 p = view_dir.xy / view_dir.z * height_scale;
	vec2 delta_tex_coords = p / num_of_layers;

	vec2 current_tex_coords = tex_coords;
	float current_depth_map_value = texture(depth_texture_sampler, current_tex_coords).r;
	while(current_layer_depth < current_depth_map_value)
	{
		current_tex_coords -= delta_tex_coords;
		current_depth_map_value = texture(depth_texture_sampler, current_tex_coords).r;
		current_layer_depth += layer_depth;
	}

	vec2 prev_tex_coords = current_tex_coords + delta_tex_coords;

	float after_depth  = current_depth_map_value - current_layer_depth;
	float before_depth = texture(depth_texture_sampler, prev_tex_coords).r - current_layer_depth + layer_depth;

	float weight = after_depth / (after_depth - before_depth);
	vec2 final_tex_coords = prev_tex_coords * weight + current_tex_coords * (1.0 - weight);

	return final_tex_coords;
}

void main(void)
{
	vec3 view_dir = normalize(tangent_space_view_position - tangent_space_fragment_world_position.xyz);
	vec2 texture_coords = parallaxMapping(pass_texture_coords, view_dir);
	if (texture_coords.x < 0.0 || texture_coords.x > 1.0 || texture_coords.y < 0.0 || texture_coords.y > 1.0)
	{
		discard;
	}

	vec3 sampled_normal = texture(normal_texture_sampler, texture_coords).rgb;
	sampled_normal = normalize(sampled_normal * 2.0 - 1.0);
	sampled_normal = TBN * sampled_normal;

	g_position = fragment_world_position.xyz;
	g_normal = sampled_normal;
	g_color_spec.rgb = texture(color_texture_sampler, texture_coords).rgb;
	g_color_spec.a = reflectivity;
}