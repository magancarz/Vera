#version 330 core

in vec3 texture_coords;

out vec4 out_Color;

uniform samplerCube skybox;

void main()
{
    out_Color = texture(skybox, texture_coords);
}
