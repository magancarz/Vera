#ifndef COMMON_HOST_DEVICE
#define COMMON_HOST_DEVICE

#ifdef __cplusplus

#include <glm/glm.hpp>
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using mat4 = glm::mat4;
using uint = unsigned int;

#else

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#endif

struct Vertex
{
    vec3 position;
    vec3 normal;
    vec2 uv;
    vec3 tangent;
};

struct ObjectDescription
{
    uint64_t vertex_address;
    uint64_t index_address;
    uint64_t material_index;
    uint64_t texture_offset;
    mat4 object_to_world;
    uint64_t padding[4];
};

struct CameraUniformBuffer
{
    mat4 proj;
    mat4 view;
};

struct PushConstantRay
{
    uint time;
    uint frames;
    uint number_of_lights;
    float weather;
    vec3 sun_position;
};

#endif