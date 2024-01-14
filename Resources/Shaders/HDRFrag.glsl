#version 330 core

in vec2 pass_texture_coords;

out vec4 out_Color;

uniform sampler2D hdr_buffer;

void main()
{
    vec3 hdr_color = texture(hdr_buffer, pass_texture_coords).rgb;
    vec3 mapped = vec3(1.0) - exp(-hdr_color * 2);
    out_Color = vec4(mapped, 1.0);
}
