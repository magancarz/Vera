#version 330 core

in vec4 world_position;
in vec2 pass_texture_coords;
in vec3 surface_normal;
in vec3 to_camera_vector;

layout (location = 0) out vec4 out_Color;

uniform sampler2D texture_sampler;

struct Light
{
	vec3 light_position;
	vec4 light_direction;
	vec3 light_color;
	vec3 attenuation;
	float cutoff_angle;
	float cutoff_angle_offset;
};

uniform Light lights[4];
uniform float reflectivity;

float brightness = 0;
float specular = 0;

void calculateSpecularValue(vec3 to_light_dir, float attenuation)
{
	vec3 halfway_dir = normalize(to_light_dir + to_camera_vector);
	specular = pow(max(dot(surface_normal, halfway_dir), 0.0), 32.0) / attenuation * reflectivity;
}

void calculateBrightnessFromDirectionalLight(int i)
{
	vec3 to_light_vector = normalize(-lights[i].light_direction.xyz);
	float n_dot1 = dot(surface_normal, to_light_vector);
	brightness = max(n_dot1, 0.0);
	calculateSpecularValue(to_light_vector, 1.0);
}

void calculateBrightnessFromPointLight(int i)
{
	vec3 to_light_vector = normalize(lights[i].light_position - world_position.xyz);
	float n_dot1 = dot(surface_normal, to_light_vector);
	float distance = length(lights[i].light_position - world_position.xyz);
	float attenuation = 1.0f / (lights[i].attenuation.x + lights[i].attenuation.y * distance + lights[i].attenuation.z * (distance * distance));
	brightness = max(n_dot1 * attenuation, 0.0);
	calculateSpecularValue(to_light_vector, attenuation);
}

void calculateBrightnessFromSpotlight(int i)
{
	vec3 to_light_vector = normalize(lights[i].light_position - world_position.xyz);
	float to_light_dot = dot(to_light_vector, normalize(-lights[i].light_direction.xyz));
	if (to_light_dot > lights[i].cutoff_angle_offset)
	{
		float n_dot1 = dot(surface_normal, to_light_vector);
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
	vec3 total_diffuse = vec3(0.0);
	vec3 total_specular = vec3(0.0);

	for(int i = 0; i < 4; ++i)
	{
		if (lights[i].cutoff_angle > 0)
		{
			calculateBrightnessFromSpotlight(i);
		}
		else if (lights[i].light_direction.w > 0)
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

	vec4 texture_color = texture(texture_sampler, pass_texture_coords);
	out_Color = vec4(total_diffuse, 1.0) * texture_color + vec4(total_specular, 1.0);
}