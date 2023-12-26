#version 330 core

in vec3 position;
in vec2 texture_coords;
in vec3 normal;

out vec4 world_position;
out vec2 pass_texture_coords;
out vec3 surface_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform vec4 plane;

void main(void) {
	world_position = model * vec4(position, 1.0);
	gl_Position = proj * view * world_position;
	pass_texture_coords = texture_coords;

	surface_normal = normalize((model * vec4(normal, 0.0)).xyz);
}