#pragma once

#include "Memory/Buffer.h"

struct ModelDescription
{
    Buffer* vertex_buffer;
    Buffer* index_buffer;
    uint64_t num_of_triangles;
    uint64_t num_of_vertices;
    uint64_t num_of_indices;
};