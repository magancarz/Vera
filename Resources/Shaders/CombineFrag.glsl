#version 330 core

out vec4 out_Color;

in vec2 pass_texture_coords;

uniform sampler2D blurred_texture;
uniform sampler2D hdr_color_buffer;

void main()
{
    vec3 blurred_color = texture(blurred_texture, pass_texture_coords).rgb;
    vec3 hdr_color = texture(hdr_color_buffer, pass_texture_coords).rgb;
    out_Color = vec4(hdr_color + blurred_color, 1.0);
}