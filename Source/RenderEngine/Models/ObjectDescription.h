#pragma once

struct ObjectDescription
{
    uint64_t vertex_address;
    uint64_t index_address;
    uint64_t material_index;
    uint64_t texture_offset;
    glm::mat4 object_to_world;
    uint64_t padding[4];
};