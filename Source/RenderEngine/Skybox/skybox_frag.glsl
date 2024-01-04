#version 330 core

in vec3 texture_coords;

out vec4 out_Color;

uniform samplerCube skybox;

void main()
{
    out_Color = vec4(vec3(texture(skybox, texture_coords).r), 1.0);
}
