#pragma once

#include "RenderEngine/AccelerationStructures/AABB.h"

struct BVHTreeNode
{
    AABB bounds;
    union
    {
        int primitives_offset;
        int second_child_offset;
    };
    uint16_t num_of_primitives;
    uint8_t axis;
    uint8_t pad[1];
};