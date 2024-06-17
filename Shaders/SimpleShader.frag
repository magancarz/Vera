#version 460

layout (location = 0) in vec2 fragment_uv;

layout (location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D image;

void main()
{
    vec3 texture_color = texture(image, fragment_uv).rgb;
    texture_color = (texture_color * (2.51 * texture_color + 0.03)) / (texture_color * (2.43 * texture_color + 0.59) + 0.14);
    out_color = vec4(texture_color, 1.0);
}