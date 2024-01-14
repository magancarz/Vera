#version 330 core

out float out_Color;

in vec2 pass_texture_coords;

uniform sampler2D ssao_input;

void main()
{
    vec2 texel_size = 1.0 / vec2(textureSize(ssao_input, 0));
    float result = 0.0;
    for (int x = -2; x < 2; ++x)
    {
        for (int y = -2; y < 2; ++y)
        {
            vec2 offset = vec2(float(x), float(y)) * texel_size;
            result += texture(ssao_input, pass_texture_coords + offset).r;
        }
    }

    out_Color = result / (4.0 * 4.0);
}
