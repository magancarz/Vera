#version 330 core

in vec2 pass_texture_coords;

out vec4 out_Color;

uniform sampler2D texture_sampler;

void main(void)
{
	out_Color = vec4(texture(texture_sampler, pass_texture_coords).rgb, 1.0);
}