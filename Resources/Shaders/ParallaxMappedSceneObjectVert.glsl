#version 330 core

struct Light
{
	vec3 light_position;
	vec3 light_color;
	vec3 attenuation;
};
const int NUM_OF_LIGHTS = 4;

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

out vec4 fragment_world_position;
out vec2 pass_texture_coords;
out vec3 view_position;
out vec3 surface_normal;
out vec3 light_positions[NUM_OF_LIGHTS];

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 light_view[NUM_OF_LIGHTS];

uniform Light lights[NUM_OF_LIGHTS];
uniform int normal_map_loaded;
uniform int lights_count;

void main(void) {
	fragment_world_position = model * vec4(position, 1.0);
	gl_Position = proj * view * fragment_world_position;
	pass_texture_coords = texture_coords;

	mat3 normal_matrix = transpose(inverse(mat3(model)));
	surface_normal = normalize(normal_matrix * normal);
	view_position = (inverse(view) * vec4(0, 0, 0, 1)).xyz;
	for (int i = 0; i < lights_count; ++i)
	{
		light_positions[i] = lights[i].light_position;
		fragment_world_position_in_light_space[i] = light_view[i] * fragment_world_position;
	}

	vec3 T = normalize(mat3(normal_matrix) * tangent);
	vec3 B = normalize(mat3(normal_matrix) * bitangent);
	mat3 TBN = transpose(mat3(T, B, surface_normal));
	fragment_world_position = vec4(TBN * vec3(fragment_world_position), 1.0);
	view_position = TBN * view_position;
	for (int i = 0; i < NUM_OF_LIGHTS; ++i)
	{
		light_positions[i] = TBN * lights[i].light_position;
		fragment_world_position_in_light_space[i] = vec4(TBN * vec3(fragment_world_position_in_light_space[i]), 1.0);
	}
}