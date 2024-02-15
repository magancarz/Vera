#version 460

layout (location = 0) in vec2 fragment_offset;

layout (location = 0) out vec4 out_color;

struct PointLight
{
    vec4 position;
    vec4 color;
};

layout(set = 0, binding = 0) uniform GlobalUbo
{
    mat4 projection_matrix;
    mat4 view_matrix;
    vec4 ambient_light_color; // w is intensity
    PointLight point_lights[10];
    int number_of_lights;
} ubo;

layout(push_constant) uniform Push
{
    vec4 position;
    vec4 color;
    float radius;
} push;

void main()
{
    float dis = sqrt(dot(fragment_offset, fragment_offset));
    if (dis >= 1.0)
    {
        discard;
    }

    out_color = vec4(push.color.xyz, 1.0);
}