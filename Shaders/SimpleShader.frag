#version 460

layout (location = 0) in vec3 normal_world_space;

layout (location = 0) out vec4 out_color;

layout(push_constant) uniform Push
{
    mat4 transform;
    mat4 normal_matrix;
} push;

const vec3 DIRECTION_TO_LIGHT = normalize(vec3(1.0, -3.0, -1.0));
const float AMBIENT = 0.02;

void main()
{
    float light_intensity = AMBIENT + max(dot(normal_world_space, DIRECTION_TO_LIGHT), 0);

    out_color = vec4(vec3(light_intensity), 1.0);
}