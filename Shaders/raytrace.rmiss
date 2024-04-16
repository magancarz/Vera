#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT Payload
{
    vec4 origin;
    vec4 direction;
    vec3 color;
    int is_active;
    uint seed;
    uint depth;
} payload;

void main()
{
    float t = 0.5 * (payload.direction.y + 1.0);
    vec3 sky_color = (1.0 - t) * vec3(1.0) + t * vec3(0.5, 0.7, 1.0);
    payload.color *= sky_color;
    payload.is_active = 0;
}