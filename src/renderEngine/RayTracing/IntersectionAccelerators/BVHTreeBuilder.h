#pragma once

#include <vector>

#include "Geometry/Bounds.h"
#include <RenderEngine/RayTracing/Shapes/Triangle.h>

#include "BVHTreeNode.h"
#include "Utils/DeviceMemoryPointer.h"
#include "BVHTreeTraverser.h"

struct ShapeInfo;

struct BVHShapeInfo
{
    BVHShapeInfo() = default;
    BVHShapeInfo(size_t shape_index, const Bounds3f& bounds)
        : shape_index(shape_index), bounds(bounds),
          centroid(.5f * bounds.min + .5f * bounds.max) {}

    size_t shape_index;
    Bounds3f bounds;
    glm::vec3 centroid;
};

struct BVHBuildNode
{
    void initializeInterior(int axis, std::shared_ptr<BVHBuildNode> left_child, std::shared_ptr<BVHBuildNode> right_child)
    {
        children[0] = std::move(left_child);
        children[1] = std::move(right_child);
        bounds = boundsFromUnion(children[0]->bounds, children[1]->bounds);
        split_axis = axis;
        num_of_shapes = 0;
    }

    void initializeLeaf(int first, int n, const Bounds3f& b)
    {
        first_shape_offset = first;
        num_of_shapes = n;
        bounds = b;
        children[0] = children[1] = nullptr;
    }

    Bounds3f bounds;
    std::shared_ptr<BVHBuildNode> children[2];
    int split_axis;
    int first_shape_offset;
    int num_of_shapes;
};

struct BucketInfo
{
    int count = 0;
    Bounds3f bounds;
};

class BVHTreeBuilder {
public:
    dmm::DeviceMemoryPointer<BVHTreeTraverser> buildAccelerator(dmm::DeviceMemoryPointer<Triangle*> shapes, dmm::DeviceMemoryPointer<ShapeInfo*> shape_infos);

private:
    std::shared_ptr<BVHBuildNode> recursiveBuild(std::vector<BVHShapeInfo>& shape_info,
        int start, int end,
        int depth,
        int* total_nodes,
        std::vector<Triangle*>& ordered_shapes);
    std::shared_ptr<BVHBuildNode> createLeafNode(
        std::shared_ptr<BVHBuildNode>& node,
        int start, int end,
        std::vector<Triangle*>& ordered_shapes,
        const std::vector<BVHShapeInfo>& shape_info,
        const Bounds3f& bounds);
    int flattenBVHTree(const std::shared_ptr<BVHBuildNode>& node, int* offset);

    dmm::DeviceMemoryPointer<BVHTreeNode> nodes;
    dmm::DeviceMemoryPointer<Triangle*> shapes;
    dmm::DeviceMemoryPointer<ShapeInfo*> shape_infos;
    size_t num_of_shapes{0};
    size_t max_shapes_in_node{0};
    size_t max_depth{0};

    dmm::DeviceMemoryPointer<BVHTreeTraverser> tree_traverser;
};
