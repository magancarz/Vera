#version 330 core

struct Light
{
	vec3 light_position;
	vec3 light_direction;
	vec3 light_color;
	vec3 attenuation;
	float cutoff_angle;
	float cutoff_angle_offset;
};
const int NUM_OF_LIGHTS = 4;

in vec4 fragment_world_position;
in vec2 pass_texture_coords;
in vec3 view_position;
in vec3 surface_normal;
in vec3 light_positions[NUM_OF_LIGHTS];

layout (location = 0) out vec4 out_Color;

uniform sampler2D color_texture_sampler;
uniform sampler2D normal_texture_sampler;
uniform sampler2D depth_texture_sampler;

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

void main(void)
{
	vec3 view_dir = normalize(view_position - fragment_world_position.xyz);
	tex_coords = pass_texture_coords;
	if (depth_map_loaded > 0)
	{
		tex_coords = parallaxMapping(pass_texture_coords, view_dir);
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

	for(int i = 0; i < NUM_OF_LIGHTS; ++i)
	{
		if (lights[i].cutoff_angle > 0)
		{
			calculateBrightnessFromSpotlight(i);
		}
		else if (lights[i].attenuation.y > 0)
		{
			calculateBrightnessFromPointLight(i);
		}
		else
		{
			calculateBrightnessFromDirectionalLight(i);
		}
		brightness = max(brightness, 0.1f);
		total_diffuse = total_diffuse + (brightness * lights[i].light_color);
		total_specular = total_specular + (specular * lights[i].light_color);
	}

	total_diffuse = max(total_diffuse, 0);

	vec4 texture_color = texture(color_texture_sampler, tex_coords);
	out_Color = vec4(total_diffuse, 1.0) * texture_color + vec4(total_specular, 1.0);
}