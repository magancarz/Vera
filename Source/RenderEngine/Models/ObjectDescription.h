#pragma once

struct ObjectDescription
{
    uint64_t vertex_address;
    uint64_t index_address;
    uint64_t material_address;
    uint64_t num_of_triangles;
    glm::mat4 object_to_world;
    alignas(16) float surface_area;
};