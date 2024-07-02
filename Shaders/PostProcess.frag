#version 460

layout (location = 0) in vec2 fragment_uv;

layout (location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D image;

vec3 adjustSaturation(vec3 color, float saturation)
{
    float grey = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(grey), color, saturation);
}

void main()
{
    vec3 texture_color = texture(image, fragment_uv).rgb;
    texture_color = adjustSaturation(texture_color, 1.4);
    out_color = vec4(texture_color, 1.0);
}