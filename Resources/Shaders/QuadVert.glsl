#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;

out vec2 pass_texture_coords;

void main(void)
{
	gl_Position = vec4(position.x, position.y, 0.0, 1.0);
	pass_texture_coords = texture_coords;
}