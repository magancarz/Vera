#version 460

layout (location = 0) out vec4 out_color;

layout(push_constant) uniform Push
{
    mat2 trasnform;
    vec2 offset;
    vec3 color;
} push;

void main()
{
    out_color = vec4(push.color, 0.0);
}