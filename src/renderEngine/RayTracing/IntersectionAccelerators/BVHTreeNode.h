#pragma once

struct BVHTreeNode
{
    Bounds3f bounds;
    union
    {
        int primitives_offset;
        int second_child_offset;
    };
    uint16_t num_of_shapes;
    uint8_t split_axis;
    uint8_t pad[1];
};