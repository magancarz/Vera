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
    uint64_t material_address;
    uint64_t num_of_triangles;
    mat4 object_to_world;
    float surface_area;
};

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 3, set = 0) buffer ObjectDescriptions { ObjectDescription data[]; } object_descriptions;
layout(binding = 4, set = 0) buffer LightIndices { uint l[]; } light_indices;

struct Vertex
{
    vec3 position;
    uint alignment1;
    vec3 normal;
    uint alignment2;
    vec2 uv;
    vec2 alignment3;
};

layout(buffer_reference, scalar) readonly buffer Vertices { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uint i[]; };
layout(buffer_reference, scalar) readonly buffer MaterialBuffer { Material m; };

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

void main()
{
    if (payload.is_active == 0)
    {
        return;
    }

    ObjectDescription object_description = object_descriptions.data[gl_InstanceCustomIndexEXT];

    MaterialBuffer material_buffer = MaterialBuffer(object_description.material_address);
    Material material = material_buffer.m;
    vec3 material_color = material.color;

    Indices index_buffer = Indices(object_description.index_address);
    uvec3 indices = uvec3(index_buffer.i[3 * gl_PrimitiveID + 0],
                        index_buffer.i[3 * gl_PrimitiveID + 1],
                        index_buffer.i[3 * gl_PrimitiveID + 2]);

    Vertices vertex_buffer = Vertices(object_description.vertex_address);
    Vertex first_vertex = vertex_buffer.v[indices.x];
    Vertex second_vertex = vertex_buffer.v[indices.y];
    Vertex third_vertex = vertex_buffer.v[indices.z];

    vec3 barycentric = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 geometric_normal = first_vertex.normal * barycentric.x + second_vertex.normal * barycentric.y + third_vertex.normal * barycentric.z;
    geometric_normal = normalize(vec3(gl_ObjectToWorldEXT * vec4(geometric_normal, 0.0)));
    geometric_normal = sign(dot(payload.direction, geometric_normal)) > 0 ? -geometric_normal : geometric_normal;

    vec3 position = first_vertex.position * barycentric.x + second_vertex.position * barycentric.y + third_vertex.position * barycentric.z;
    position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

    payload.origin = position;

    ObjectDescription random_light = object_descriptions.data[light_indices.l[uint(floor(rnd(payload.seed) * float(push_constant.number_of_lights)))]];
    Indices light_index_buffer = Indices(random_light.index_address);

    uint light_random_primitive = uint(floor(rnd(payload.seed) * float(random_light.num_of_triangles)));

    uvec3 light_indices = uvec3(light_index_buffer.i[3 * light_random_primitive + 0],
                                light_index_buffer.i[3 * light_random_primitive + 1],
                                light_index_buffer.i[3 * light_random_primitive + 2]);

    Vertices light_vertices = Vertices(random_light.vertex_address);
    Vertex light_first_vertex = light_vertices.v[light_indices.x];
    Vertex light_second_vertex = light_vertices.v[light_indices.y];
    Vertex light_third_vertex = light_vertices.v[light_indices.z];

    vec2 uv = vec2(rnd(payload.seed), rnd(payload.seed));
    if (uv.x + uv.y > 1.0f)
    {
        uv.x = 1.0f - uv.x;
        uv.y = 1.0f - uv.y;
    }

    vec3 light_barycentric = vec3(1.0 - uv.x - uv.y, uv.x, uv.y);
    vec3 light_normal = light_first_vertex.normal * light_barycentric.x + light_second_vertex.normal * light_barycentric.y + light_third_vertex.normal * light_barycentric.z;
    light_normal = normalize(light_normal);

    vec3 light_position = light_first_vertex.position * light_barycentric.x + light_second_vertex.position * light_barycentric.y + light_third_vertex.position * light_barycentric.z;
    light_position = vec3(random_light.object_to_world * vec4(light_position, 1.0));

    vec3 random_dir = randomUnitDirection();
    random_dir *= sign(dot(random_dir, push_constant.sun_position));
    vec3 positionToLightDirection = normalize(push_constant.sun_position + random_dir * push_constant.weather);

    vec3 u, v, w;
    w = geometric_normal;
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

    float scattering_pdf = scatteringPDFFromLambertian(payload.direction, geometric_normal);
    float cosine = occluded ? 0 : max(dot(push_constant.sun_position, payload.direction), 0.0);
    vec3 sun_contribution = vec3(1) * 10 * cosine;

    payload.color *= sun_contribution * material_color * scattering_pdf;
    payload.depth += 1;
}
