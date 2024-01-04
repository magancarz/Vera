#version 330 core

struct Light
{
	int light_type;
	vec3 light_position;
	vec3 light_direction;
	vec3 light_color;
	vec3 attenuation;
	float cutoff_angle;
	float cutoff_angle_offset;
};
const int NUM_OF_LIGHTS = 4;
const float ambient = 0.1;

in vec4 fragment_world_position;
in vec4 fragment_world_position_in_light_space[NUM_OF_LIGHTS];
in vec2 pass_texture_coords;
in vec3 view_position;
in vec3 surface_normal;
in vec3 light_positions[NUM_OF_LIGHTS];

layout (location = 0) out vec4 out_Color;

uniform sampler2D color_texture_sampler;
uniform sampler2D normal_texture_sampler;
uniform sampler2D depth_texture_sampler;
uniform sampler2D shadow_map_texture_sampler[NUM_OF_LIGHTS];
uniform samplerCube shadow_cube_map_texture_sampler[NUM_OF_LIGHTS];

uniform int lights_count;
uniform Light lights[NUM_OF_LIGHTS];

uniform float reflectivity;
uniform float height_scale;

uniform int normal_map_loaded;
uniform int depth_map_loaded;

float brightness = 0;
float specular = 0;
vec3 normal;
vec2 tex_coords;

void calculateSpecularValue(vec3 to_light_dir, float attenuation)
{
	vec3 to_camera_vector = normalize(view_position - fragment_world_position.xyz);
	vec3 halfway_dir = normalize(to_light_dir + to_camera_vector);
	specular = pow(max(dot(normal, halfway_dir), 0.0), 32.0) / attenuation * reflectivity;
}

void calculateBrightnessFromDirectionalLight(int i)
{
	vec3 to_light_vector = normalize(-lights[i].light_direction);
	float n_dot1 = dot(normal, to_light_vector);
	brightness = max(n_dot1, 0.0);
	calculateSpecularValue(to_light_vector, 1.0);
}

void calculateBrightnessFromPointLight(int i)
{
	vec3 to_light_vector = normalize(light_positions[i] - fragment_world_position.xyz);
	float n_dot1 = dot(normal, to_light_vector);
	float distance = length(light_positions[i] - fragment_world_position.xyz);
	float attenuation = 1.0f / (lights[i].attenuation.x + lights[i].attenuation.y * distance + lights[i].attenuation.z * (distance * distance));
	brightness = max(n_dot1 * attenuation, 0.0);
	calculateSpecularValue(to_light_vector, attenuation);
}

void calculateBrightnessFromSpotlight(int i)
{
	vec3 to_light_vector = normalize(lights[i].light_position - fragment_world_position.xyz);
	float to_light_dot = dot(to_light_vector, normalize(-lights[i].light_direction));
	if (to_light_dot > lights[i].cutoff_angle_offset)
	{
		float n_dot1 = dot(normal, to_light_vector);
		float distance = length(lights[i].light_position - fragment_world_position.xyz);
		float attenuation = 1.0f / (lights[i].attenuation.x + lights[i].attenuation.y * distance + lights[i].attenuation.z * (distance * distance));
		brightness = max(n_dot1, 0.0);
		float intensity = clamp((to_light_dot - lights[i].cutoff_angle_offset) / (lights[i].cutoff_angle - lights[i].cutoff_angle_offset), 0.0, 1.0);
		brightness *= intensity * attenuation;
		calculateSpecularValue(to_light_vector, attenuation);
		return;
	}

	brightness = 0;
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

float pointLightShadowCalculation(int light_index, int texture_index)
{
	vec3 fragment_to_light = fragment_world_position.xyz - lights[light_index].light_position;
	float closest_depth = texture(shadow_cube_map_texture_sampler[texture_index], fragment_to_light).r;
	closest_depth *= 30.0;
	float current_depth = length(fragment_to_light);
	float bias = 0.05;
	float shadow = current_depth - bias > closest_depth ? 1.0 : 0.0;

	return shadow;
}

float directionalLightShadowCalculation(int light_index, int texture_index)
{
	return 0.0;

	vec3 proj_coords = fragment_world_position_in_light_space[light_index].xyz / fragment_world_position_in_light_space[light_index].w;
	proj_coords = proj_coords * 0.5 + 0.5;

	float shadow = 0.0;
	float closest_depth = texture(shadow_map_texture_sampler[texture_index], proj_coords.xy).r;
	float current_depth = proj_coords.z;
	float bias = max(0.005 * (1.0 - dot(normal, vec3(0, -1, 0))), 0.0005);
	vec2 texel_size = 1.0 / textureSize(shadow_map_texture_sampler[texture_index], 0);
	for (int x = -1; x <= 1; ++x)
	{
		for (int y = -1; y <= 1; ++y)
		{
			float pcf_depth = texture(shadow_map_texture_sampler[texture_index], proj_coords.xy + vec2(x, y) * texel_size).r;
			shadow += current_depth - bias > pcf_depth ? 1.0 : 0.0;
		}
	}
	shadow /= 9.0;

	if (proj_coords.z > 1.0)
	{
		shadow =  0.0;
	}

	return shadow;
}

void main(void)
{
	vec3 view_dir = normalize(view_position - fragment_world_position.xyz);
	tex_coords = pass_texture_coords;
	if (depth_map_loaded > 0)
	{
		tex_coords = parallaxMapping(pass_texture_coords, view_dir);
		if (tex_coords.x < 0 || tex_coords.x > 1.0 || tex_coords.y < 0 || tex_coords.y > 1.0)
		{
			discard;
		}
	}

	if (normal_map_loaded > 0)
	{
		normal = texture(normal_texture_sampler, tex_coords).rgb;
		normal = normalize(normal * 2.0 - 1.0);
	}
	else
	{
		normal = surface_normal;
	}

	vec3 total_diffuse = vec3(0.0);
	vec3 total_specular = vec3(0.0);

	int shadow_map_index = 0;
	int cube_shadow_map_index = 0;
	float total_shadow = 0.0;

	for(int i = 0; i < lights_count; ++i)
	{
		switch(lights[i].light_type)
		{
			case 0:
				calculateBrightnessFromDirectionalLight(i);
				total_shadow += directionalLightShadowCalculation(i, shadow_map_index);
				++shadow_map_index;
			break;
			case 1:
				calculateBrightnessFromPointLight(i);
				total_shadow += pointLightShadowCalculation(i, cube_shadow_map_index);
				++cube_shadow_map_index;
			break;
			case 2:
				calculateBrightnessFromSpotlight(i);
				total_shadow += directionalLightShadowCalculation(i, shadow_map_index);
				++shadow_map_index;
			break;
		}

		total_diffuse = total_diffuse + (brightness * lights[i].light_color);
		total_specular = total_specular + (specular * lights[i].light_color);
	}
	
	total_diffuse = max(total_diffuse, 0);
	total_specular = max(total_specular, 0);
	total_shadow /= lights_count;

	vec4 texture_color = texture(color_texture_sampler, tex_coords);
	vec3 lighting = (ambient + (1.0 - total_shadow) * (total_diffuse + total_specular)) * texture_color.rgb;
	out_Color = vec4(lighting, 1.0);
}