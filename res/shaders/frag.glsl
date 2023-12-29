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

in vec4 world_position;
in vec2 pass_texture_coords;
in vec3 to_camera_vector;
in vec3 surface_normal;
in mat3 TBN;

layout (location = 0) out vec4 out_Color;

uniform sampler2D color_texture_sampler;
uniform sampler2D normal_texture_sampler;

uniform Light lights[NUM_OF_LIGHTS];
uniform float reflectivity;
uniform int normal_map_loaded;

float brightness = 0;
float specular = 0;
vec3 normal;

void calculateSpecularValue(vec3 to_light_dir, float attenuation)
{
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
	vec3 to_light_vector = normalize(lights[i].light_position - world_position.xyz);
	float n_dot1 = dot(normal, to_light_vector);
	float distance = length(lights[i].light_position - world_position.xyz);
	float attenuation = 1.0f / (lights[i].attenuation.x + lights[i].attenuation.y * distance + lights[i].attenuation.z * (distance * distance));
	brightness = max(n_dot1 * attenuation, 0.0);
	calculateSpecularValue(to_light_vector, attenuation);
}

void calculateBrightnessFromSpotlight(int i)
{
	vec3 to_light_vector = normalize(lights[i].light_position - world_position.xyz);
	float to_light_dot = dot(to_light_vector, normalize(-lights[i].light_direction));
	if (to_light_dot > lights[i].cutoff_angle_offset)
	{
		float n_dot1 = dot(normal, to_light_vector);
		float distance = length(lights[i].light_position - world_position.xyz);
		float attenuation = 1.0f / (lights[i].attenuation.x + lights[i].attenuation.y * distance + lights[i].attenuation.z * (distance * distance));
		brightness = max(n_dot1, 0.0);
		float intensity = clamp((to_light_dot - lights[i].cutoff_angle_offset) / (lights[i].cutoff_angle - lights[i].cutoff_angle_offset), 0.0, 1.0);
		brightness *= intensity * attenuation;
		calculateSpecularValue(to_light_vector, attenuation);
		return;
	}

	brightness = 0;
}

void main(void)
{
	if (normal_map_loaded > 0)
	{
		normal = texture(normal_texture_sampler, pass_texture_coords).rgb;
		normal = normalize(normal * 2.0 - 1.0);
		normal = TBN * normal;
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

	vec4 texture_color = texture(color_texture_sampler, pass_texture_coords);
	out_Color = vec4(total_diffuse, 1.0) * texture_color + vec4(total_specular, 1.0);
}