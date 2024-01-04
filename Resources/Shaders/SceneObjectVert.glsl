#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;
layout (location = 2) in vec3 normal;

out vec4 fragment_world_position;
out vec2 pass_texture_coords;
out vec3 view_position;
out vec3 surface_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main(void)
{
	fragment_world_position = model * vec4(position, 1.0);
	gl_Position = proj * view * fragment_world_position;
	pass_texture_coords = texture_coords;

	surface_normal = normalize(mat3(model) * normal);
	view_position = (inverse(view) * vec4(0, 0, 0, 1)).xyz;
}