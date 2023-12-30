#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;
layout (location = 2) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

const float density = 0.002;
const float gradient = 1.5;

void main(void) {
	vec4 world_position = model * vec4(position, 1.0);

	vec4 position_relative_to_cam = view * world_position;
	gl_Position = proj * view * model * vec4(position, 1.0);
}