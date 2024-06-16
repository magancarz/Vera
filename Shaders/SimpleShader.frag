#version 460

layout (location = 0) in vec2 fragment_uv;

layout (location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D image;

mat3 ACESInputMat =
{
    {0.59719, 0.35458, 0.04823},
    {0.07600, 0.90834, 0.01566},
    {0.02840, 0.13383, 0.83777}
};

mat3 ACESOutputMat =
{
    { 1.60475, -0.53108, -0.07367},
    {-0.10208,  1.10813, -0.00605},
    {-0.00327, -0.07276,  1.07602}
};

vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

void main()
{
    vec3 texture_color = texture(image, fragment_uv).rgb;
//    texture_color = texture_color * ACESInputMat;
//    texture_color = RRTAndODTFit(texture_color);
//    texture_color = texture_color * ACESOutputMat;

    texture_color = (texture_color*(2.51f*texture_color+0.03f))/(texture_color*(2.43f*texture_color+0.59f)+0.14f);

    out_color = vec4(texture_color, 1.0);
}