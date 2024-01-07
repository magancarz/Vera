#version 330 core

in vec4 fragment_world_position;
in vec2 pass_texture_coords;
in vec3 view_position;
in vec3 surface_normal;

layout (location = 0) out vec4 out_Color;

uniform sampler2D color_texture_sampler;

uniform float reflectivity;

struct Light
{
	vec3 light_position;
	vec3 light_color;
	vec3 attenuation;
};
const int NUM_OF_LIGHTS = 4;
const int lights_count = 1;
const float ambient = 0.1;

layout (std140) uniform LightInfos
{
	Light light;
};

uniform samplerCube shadow_map;

float calculateSpecularValue(vec3 to_light_dir, float attenuation)
{
	vec3 normalized_to_light_dir = normalize(to_light_dir);
	vec3 to_camera_vector = normalize(view_position - fragment_world_position.xyz);
	vec3 halfway_dir = normalize(normalized_to_light_dir + to_camera_vector);
	float specular = pow(max(dot(surface_normal, halfway_dir), 0.0), 32.0) / attenuation * reflectivity;

	return specular;
}

float calculateBrightnessFromPointLight(int i, vec3 to_light_dir, float attenuation)
{
	vec3 normalized_to_light_dir = normalize(to_light_dir);
	float n_dot1 = dot(surface_normal, normalized_to_light_dir);
	float brightness = max(n_dot1 * attenuation, 0.0);

	return brightness;
}

float pointLightShadowCalculation(int light_index, int texture_index)
{
	vec3 fragment_to_light = fragment_world_position.xyz - light.light_position;
	float closest_depth = texture(shadow_map, fragment_to_light).r;
	closest_depth *= 30.0;
	float current_depth = length(fragment_to_light);
	float bias = 0.05;
	float shadow = current_depth - bias > closest_depth ? 1.0 : 0.0;

	return shadow;
}

void main(void)
{
	vec3 view_dir = normalize(view_position - fragment_world_position.xyz);

	vec3 total_diffuse = vec3(0.0);
	vec3 total_specular = vec3(0.0);

	int cube_shadow_map_index = 0;
	float total_shadow = 0.0;

	for(int i = 0; i < 1; ++i)
	{
		vec3 to_light_vector = light.light_position - fragment_world_position.xyz;
		float distance = length(to_light_vector);
		float attenuation = 1.0f / (light.attenuation.x + light.attenuation.y * distance + light.attenuation.z * (distance * distance));
		float brightness = calculateBrightnessFromPointLight(i, to_light_vector, attenuation);
		float specular = calculateSpecularValue(to_light_vector, attenuation);
		total_shadow += pointLightShadowCalculation(i, i);

		total_diffuse = total_diffuse + (brightness * light.light_color);
		total_specular = total_specular + (specular * light.light_color);
	}
	
	total_diffuse = max(total_diffuse, 0);
	total_specular = max(total_specular, 0);
	total_shadow /= lights_count;

	vec4 texture_color = texture(color_texture_sampler, pass_texture_coords);
	vec3 lighting = (ambient + (1.0 - total_shadow) * (total_diffuse + total_specular));
	vec3 final_color = lighting * texture_color.rgb;
	out_Color = vec4(final_color, 1.0);
}