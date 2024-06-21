#pragma once

#include "glm/vec4.hpp"

struct OctreeNode
{
    glm::vec4 children_and_color{0};
    glm::vec4 aabb_min{0};
    glm::vec4 aabb_max{0};
    uint32_t child_offsets[8];
};
