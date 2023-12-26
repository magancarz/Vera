#version 330 core

in vec4 world_position;
in vec2 pass_texture_coords;
in vec3 surface_normal;

layout (location = 0) out vec4 out_Color;

uniform sampler2D texture_sampler;

struct Light
{
	vec3 light_position;
	vec4 light_direction;
	vec3 light_color;
	vec3 attenuation;
	float cutoff_angle;
};

uniform Light lights[4];

float calculateBrightnessFromDirectionalLight(int i)
{
	vec3 to_light_vector = normalize(-lights[i].light_direction.xyz);
	float n_dot1 = dot(surface_normal, to_light_vector);
	float brightness = max(n_dot1, 0.0);
	return brightness;
}

float calculateBrightnessFromPointLight(int i)
{
	vec3 to_light_vector = normalize(lights[i].light_position - world_position.xyz);
	float n_dot1 = dot(surface_normal, to_light_vector);
	float distance = length(lights[i].light_position - world_position.xyz);
	float attenuation = 1.0f / (lights[i].attenuation.x + lights[i].attenuation.y * distance + lights[i].attenuation.z * (distance * distance));
	float brightness = max(n_dot1 * attenuation, 0.0);
	return brightness;
}

float calculateBrightnessFromSpotlight(int i)
{
	vec3 to_light_vector = normalize(lights[i].light_position - world_position.xyz);
	float to_light_dot = dot(to_light_vector, normalize(-lights[i].light_direction.xyz));
	if (to_light_dot > lights[i].cutoff_angle)
	{
		float n_dot1 = dot(surface_normal, to_light_vector);
		float distance = length(lights[i].light_position - world_position.xyz);
		float attenuation = 1.0f / (lights[i].attenuation.x + lights[i].attenuation.y * distance + lights[i].attenuation.z * (distance * distance));
		float brightness = max(n_dot1, 0.0) * attenuation;
		return brightness;
	}

	return 0;
}

void main(void)
{
	vec3 total_diffuse = vec3(0.0);

	for(int i = 0; i < 4; ++i)
	{
		float brightness = 0;
		if (lights[i].cutoff_angle > 0)
		{
			brightness = calculateBrightnessFromSpotlight(i);
		}
		else if (lights[i].light_direction.w > 0)
		{
			brightness = calculateBrightnessFromPointLight(i);
		}
		else
		{
			brightness = calculateBrightnessFromDirectionalLight(i);
		}
		brightness = max(brightness, 0.1f);
		total_diffuse = total_diffuse + (brightness * lights[i].light_color);
	}

	total_diffuse = max(total_diffuse, 0);

	vec4 texture_color = texture(texture_sampler, pass_texture_coords);
	out_Color = vec4(total_diffuse, 1.0) * texture_color;
}