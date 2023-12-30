#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;
layout (location = 2) in vec3 normal;

out vec2 pass_texture_coords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

const float density = 0.002;
const float gradient = 1.5;

void main(void)
{
	gl_Position = vec4(position.x, position.y, 0.0, 1.0);
	pass_texture_coords = texture_coords;
}