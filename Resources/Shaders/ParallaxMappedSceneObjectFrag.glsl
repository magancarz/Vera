#version 330 core

struct Light
{
	vec3 light_position;
	vec3 light_color;
	vec3 attenuation;
	samplerCube shadow_map;
};
const int NUM_OF_LIGHTS = 4;
layout (std140) uniform Lights
{
	int lights_count;
	Light lights[NUM_OF_LIGHTS];
};

in vec4 fragment_world_position;
in vec4 tangent_space_fragment_world_position;
in vec2 pass_texture_coords;
in vec3 tangent_space_view_position;
in vec3 tangent_space_light_positions[NUM_OF_LIGHTS];

layout (location = 0) out vec4 out_Color;

uniform sampler2D color_texture_sampler;
uniform sampler2D normal_texture_sampler;
uniform sampler2D depth_texture_sampler;

uniform float reflectivity;
uniform float height_scale;

float calculateSpecularValue(vec3 to_light_dir, float attenuation, vec3 sampled_normal)
{
	vec3 normalized_to_light_dir = normalize(to_light_dir);
	vec3 to_camera_vector = normalize(tangent_space_view_position - tangent_space_fragment_world_position.xyz);
	vec3 halfway_dir = normalize(normalized_to_light_dir + to_camera_vector);
	float specular = pow(max(dot(sampled_normal, halfway_dir), 0.0), 32.0) / attenuation * reflectivity;

	return specular;
}

float calculateBrightnessFromPointLight(int i, vec3 to_light_dir, float attenuation, vec3 sampled_normal)
{
	vec3 normalized_to_light_dir = normalize(to_light_dir);
	float n_dot1 = dot(sampled_normal, normalized_to_light_dir);
	float brightness = max(n_dot1 * attenuation, 0.0);

	return brightness;
}

float pointLightShadowCalculation(int light_index, int texture_index)
{
	vec3 fragment_to_light = fragment_world_position.xyz - lights[light_index].light_position;
	float closest_depth = texture(shadow_cube_map_texture_sampler[texture_index], fragment_to_light).r;
	closest_depth *= 30.0;
	float current_depth = length(fragment_to_light);
	float bias = 0.05;
	float shadow = closest_depth - bias > closest_depth ? 1.0 : 0.0;

	return shadow;
}

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

	vec3 total_diffuse = vec3(0.0);
	vec3 total_specular = vec3(0.0);

	float total_shadow = 0.0;

	for(int i = 0; i < lights_count; ++i)
	{
		vec3 to_light_vector = tangent_space_light_positions[i] - tangent_space_fragment_world_position.xyz;
		float distance = length(to_light_vector);
		float attenuation = 1.0f / (lights[i].attenuation.x + lights[i].attenuation.y * distance + lights[i].attenuation.z * (distance * distance));
		float brightness = calculateBrightnessFromPointLight(i, to_light_vector, attenuation, sampled_normal);
		float specular = calculateSpecularValue(to_light_vector, attenuation, sampled_normal);
		total_shadow += pointLightShadowCalculation(i, i);

		total_diffuse = total_diffuse + (brightness * lights[i].light_color);
		total_specular = total_specular + (specular * lights[i].light_color);
	}

	total_diffuse = max(total_diffuse, 0);
	total_specular = max(total_specular, 0);
	total_shadow /= lights_count;

	vec4 texture_color = texture(color_texture_sampler, texture_coords);
	vec3 lighting = (ambient + (1.0 - total_shadow) * (total_diffuse + total_specular)) * texture_color.rgb;
	out_Color = vec4(lighting, 1.0);
}