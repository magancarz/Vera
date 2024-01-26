#version 330 core

in vec2 pass_texture_coords;

layout (location = 0) out vec4 out_Color;
layout (location = 1) out vec4 out_BrightColor;

uniform sampler2D hdr_color_buffer;

void main()
{
    vec3 color = texture(hdr_color_buffer, pass_texture_coords).rgb;
    out_Color = vec4(color, 1.0);
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > 1.0)
    {
        out_BrightColor = out_Color;
    }
}
