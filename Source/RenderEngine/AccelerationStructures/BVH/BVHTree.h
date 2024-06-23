#pragma once

#include <memory>
#include <vector>

#include "Assets/Model/Model.h"
#include "RenderEngine/AccelerationStructures/AABB.h"

struct BVHPrimitiveInfo
{
    BVHPrimitiveInfo() = default;
    BVHPrimitiveInfo(size_t primitive_num, const AABB& bounds)
        : primitive_num(primitive_num), bounds(bounds), centroid(.5f * bounds.min + .5f * bounds.max) {}

    size_t primitive_num;
    AABB bounds;
    glm::vec3 centroid;
};

struct BVHBuildNode
{
    void initializeInterior(int axis, std::unique_ptr<BVHBuildNode> left_child, std::unique_ptr<BVHBuildNode> right_child)
    {
        children[0] = std::move(left_child);
        children[1] = std::move(right_child);
        bounds = AABB::merge(children[0]->bounds, children[1]->bounds);
        split_axis = axis;
    }

    void initializeLeaf(int first, int n, const AABB& b)
    {
        first_shape_offset = first;
        num_of_shapes = n;
        bounds = b;
    }

    AABB bounds;
    std::unique_ptr<BVHBuildNode> children[2];
    int split_axis{0};
    int first_shape_offset{0};
    int num_of_shapes{0};
};

class BVHTree {
public:
    BVHTree(const Model* model, uint32_t max_prims_in_node, uint32_t max_depth = 8);

protected:
    const Model* model;
    uint32_t max_prims_in_node{0};
    uint32_t max_depth{0};
    uint32_t num_of_shapes{0};

    void buildAccelerator();

    std::unique_ptr<BVHBuildNode> recursiveBuild(
        std::vector<BVHPrimitiveInfo>& primitive_info, int start, int end, uint32_t current_depth, std::vector<uint32_t>& ordered_shapes);

    std::unique_ptr<BVHBuildNode> root;
};