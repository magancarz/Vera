#version 330 core

out vec4 out_Color;

uniform vec3 light_color;

void main(void)
{
    out_Color = vec4(light_color, 1.0);
}