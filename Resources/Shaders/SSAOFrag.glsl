#version 330 core

out float out_Color;

in vec2 pass_texture_coords;

uniform sampler2D g_position;
uniform sampler2D g_normal;
uniform sampler2D noise_texture;

uniform vec3 samples[64];

layout (std140) uniform TransformationMatrices
{
    mat4 model;
    mat4 view;
    mat4 proj;
};

const vec2 noise_scale = vec2(1280.0/4.0, 800.0/4.0);
const int kernel_size = 64;
const float radius = 0.5;
const float bias = 0.025;

void main()
{
    vec3 fragment_position = texture(g_position, pass_texture_coords).xyz;
    vec3 normal = texture(g_normal, pass_texture_coords).rgb;
    vec3 random_vector = texture(noise_texture, pass_texture_coords * noise_scale).xyz;

    vec3 tangent = normalize(random_vector - normal * dot(random_vector, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for(int i = 0; i < kernel_size; ++i)
    {
        vec3 sample_position = TBN * samples[i];
        sample_position = fragment_position + sample_position * radius;

        vec4 offset = vec4(sample_position, 1.0);
        offset = proj * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sample_depth = texture(g_position, offset.xy).z;

        float range_check = smoothstep(0.0, 1.0, radius / abs(fragment_position.z - sample_depth));
        occlusion += (sample_depth >= sample_position.z + bias ? 1.0 : 0.0) * range_check;
    }
    occlusion = 1.0 - (occlusion / kernel_size);
    out_Color = pow(occlusion, 2);
}
