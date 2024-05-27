#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "ray.glsl"
#include "random.glsl"
#include "cosine_pdf.glsl"
#include "lambertian.glsl"
#include "material.glsl"
#include "defines.glsl"

layout(location = 0) rayPayloadInEXT Ray payload;
layout(location = 1) rayPayloadEXT bool occluded;

struct ObjectDescription
{
    uint64_t vertex_address;
    uint64_t index_address;
    uint64_t material_index;
    uint64_t num_of_triangles;
    uint64_t texture_offset;
    mat4 object_to_world;
    float surface_area;
};

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 3, set = 0) buffer ObjectDescriptions { ObjectDescription data[]; } object_descriptions;
layout(binding = 4, set = 0) buffer Materials { Material m[]; } materials;
layout(binding = 5, set = 0) uniform sampler2D diffuse_textures[];
layout(binding = 6, set = 0) uniform sampler2D normal_textures[];

struct Vertex
{
    vec3 position;
    uint alignment1;
    vec3 normal;
    uint alignment2;
    vec2 uv;
    vec2 alignment3;
    vec3 tangent;
    uint alignment4;
    vec3 bitangent;
    uint alignment5;
};

layout(buffer_reference, scalar) readonly buffer Vertices { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uint i[]; };

layout(push_constant) uniform PushConstantRay
{
    uint time;
    uint frames;
    uint number_of_lights;
    float weather;
    vec3 sun_position;
} push_constant;

hitAttributeEXT vec3 attribs;

vec3 randomUnitDirection()
{
    float r1 = rnd(payload.seed);
    float r2 = rnd(payload.seed);
    float x = cos(2 * PI * r1) * 2 * sqrt(r2 * (1 - r2));
    float y = sin(2 * PI * r1) * 2 * sqrt(r2 * (1 - r2));
    float z = 1 - 2 * r2;
    return vec3(x, y, z);
}

vec3 randomToSphere(float radius, float distance_squared)
{
    float r1 = rnd(payload.seed);
    float r2 = rnd(payload.seed);
    float z = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);

    float phi = 2 * PI * r1;
    float x = cos(phi) * sqrt(1 - z * z);
    float y = sin(phi) * sqrt(1 - z * z);

    return vec3(x, y, z);
}

vec3 randomToSun(float radius)
{
    const float distance_squared = 100.0;
    vec3 u, v, w;
    w = push_constant.sun_position;
    generateOrthonormalBasis(u, v, w);
    vec3 random_to_sphere = randomToSphere(radius, distance_squared);
    random_to_sphere = u * random_to_sphere.x + v * random_to_sphere.y + w * random_to_sphere.z;
    return random_to_sphere;
}

void main()
{
    if (payload.is_active == 0)
    {
        return;
    }

    ObjectDescription object_description = object_descriptions.data[gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT];
    Material material = materials.m[uint(object_description.material_index)];

    Indices index_buffer = Indices(object_description.index_address);
    uvec3 indices = uvec3(index_buffer.i[3 * gl_PrimitiveID + 0],
                        index_buffer.i[3 * gl_PrimitiveID + 1],
                        index_buffer.i[3 * gl_PrimitiveID + 2]);

    Vertices vertex_buffer = Vertices(object_description.vertex_address);
    Vertex first_vertex = vertex_buffer.v[indices.x];
    Vertex second_vertex = vertex_buffer.v[indices.y];
    Vertex third_vertex = vertex_buffer.v[indices.z];

    vec3 barycentric = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 position = first_vertex.position * barycentric.x + second_vertex.position * barycentric.y + third_vertex.position * barycentric.z;
    position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

    vec2 texture_uv = first_vertex.uv * barycentric.x + second_vertex.uv * barycentric.y + third_vertex.uv * barycentric.z;

    vec3 edge1 = second_vertex.position - first_vertex.position;
    vec3 edge2 = third_vertex.position - first_vertex.position;
    vec2 delta_UV1 = second_vertex.uv - first_vertex.uv;
    vec2 delta_UV2 = third_vertex.uv - first_vertex.uv;

    vec3 tangent, bitangent;
    float f = 1.0f / (delta_UV1.x * delta_UV2.y - delta_UV2.x * delta_UV1.y);
    tangent.x = f * (delta_UV2.y * edge1.x - delta_UV1.y * edge2.x);
    tangent.y = f * (delta_UV2.y * edge1.y - delta_UV1.y * edge2.y);
    tangent.z = f * (delta_UV2.y * edge1.z - delta_UV1.y * edge2.z);
    tangent = normalize(tangent);

    bitangent.x = f * (-delta_UV2.x * edge1.x + delta_UV1.x * edge2.x);
    bitangent.y = f * (-delta_UV2.x * edge1.y + delta_UV1.x * edge2.y);
    bitangent.z = f * (-delta_UV2.x * edge1.z + delta_UV1.x * edge2.z);
    bitangent = normalize(bitangent);

    mat3 normal_matrix = transpose(inverse(mat3(gl_ObjectToWorldEXT)));
    vec3 geometric_normal = first_vertex.normal * barycentric.x + second_vertex.normal * barycentric.y + third_vertex.normal * barycentric.z;
    geometric_normal = normalize(normal_matrix * geometric_normal);
    geometric_normal = -sign(dot(payload.direction, geometric_normal)) * geometric_normal;

    vec3 T = normalize(normal_matrix * tangent);
    vec3 B = normalize(normal_matrix * bitangent);
    vec3 N = geometric_normal;
    mat3 TBN = mat3(T, B, N);

    uint normal_texture_offset = uint(material.normal_texture_offset);
    vec3 texture_normal = texture(normal_textures[nonuniformEXT(normal_texture_offset)], texture_uv).xyz;
    vec3 normal = texture_normal * 2.0 - 1.0;
    normal = normalize(TBN * normal);

    payload.origin = position;

    vec3 positionToLightDirection = randomToSun(push_constant.weather * 10);

    vec3 u, v, w;
    w = normal;
    generateOrthonormalBasis(u, v, w);
    vec3 random_cosine_direction = generateRandomDirectionWithCosinePDF(payload.seed, u, v, w);
    payload.direction = rnd(payload.seed) > 0.5 ? positionToLightDirection : random_cosine_direction;

    const uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
    occluded = false;
    traceRayEXT(
        topLevelAS, // acceleration structure
        rayFlags,       // rayFlags
        0xFF,           // cullMask
        1,              // sbtRecordOffset
        0,              // sbtRecordStride
        1,              // missIndex
        payload.origin,     // ray origin
        T_MIN,           // ray min range
        payload.direction,  // ray direction
        T_MAX,           // ray max range
        1               // payload (location = 0)
        );

    float scattering_pdf = scatteringPDFFromLambertian(payload.direction, normal);
    float cosine = occluded ? 0 : max(dot(push_constant.sun_position, payload.direction), 0.0);
    float sun_contribution = 35 * cosine;

    uint texture_offset = uint(material.diffuse_texture_offset);
    vec3 texture_color = texture(diffuse_textures[nonuniformEXT(texture_offset)], texture_uv).xyz;
    payload.color *= sun_contribution * texture_color * scattering_pdf;
    payload.depth += 1;
}
