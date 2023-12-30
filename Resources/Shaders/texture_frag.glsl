#version 330 core

in vec2 pass_texture_coords;

out vec4 out_Color;

uniform sampler2D texture_sampler;

void main(void)
{
	//out_Color = texture(texture_sampler, pass_texture_coords);

	float depthValue = texture(texture_sampler, pass_texture_coords).r;
	out_Color = vec4(vec3(depthValue), 1.0);
}