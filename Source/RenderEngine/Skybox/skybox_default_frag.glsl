#version 330 core

in vec3 texture_coords;

out vec4 out_Color;

void main()
{
    float t = 0.5 * (texture_coords.y + 1.0);
    vec3 color = (1.0 - t) * vec3(1) + t * vec3(0.5, 0.7, 1.0);
    out_Color = vec4(color, 1.0);
}
