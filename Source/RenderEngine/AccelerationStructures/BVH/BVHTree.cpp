#include "BVHTree.h"

#include <chrono>

#include "Assets/Model/Vertex.h"
#include "Logs/LogSystem.h"

BVHTree::BVHTree(const Model* model, uint32_t max_prims_in_node, uint32_t max_depth)
    : model{model}, max_prims_in_node{max_prims_in_node}, max_depth{max_depth}
{
    buildAccelerator();
}

void BVHTree::buildAccelerator()
{
    auto start = std::chrono::high_resolution_clock::now();

    num_of_shapes = model->indices.size() / 3;

    std::vector<BVHPrimitiveInfo> primitive_info(num_of_shapes);
    for (size_t i = 0; i < num_of_shapes; ++i)
    {
        AABB triangle_aabb = AABB::fromTriangle(
            model->vertices[model->indices[i * 3]], model->vertices[model->indices[i * 3 + 1]], model->vertices[model->indices[i * 3 + 2]]);
        primitive_info[i] = {i, triangle_aabb};
    }

    std::vector<uint32_t> ordered_shapes;
    root = recursiveBuild(primitive_info, 0, num_of_shapes, 0, ordered_shapes);

    auto end = std::chrono::high_resolution_clock::now();
    auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    LogSystem::log(LogSeverity::LOG, "Built BVH tree from ", num_of_shapes, " triangles in ", delta_time, " millis.");
}

std::unique_ptr<BVHBuildNode> BVHTree::recursiveBuild(std::vector<BVHPrimitiveInfo>& primitive_info, int start, int end, uint32_t current_depth, std::vector<uint32_t>& ordered_shapes)
{
    auto node = std::make_unique<BVHBuildNode>();
    AABB bounds;
    for (int i = start; i < end; ++i)
    {
        bounds = AABB::merge(bounds, primitive_info[i].bounds);
    }

    const size_t num_of_primitives = end - start;
    if (num_of_primitives <= max_prims_in_node || current_depth >= max_depth)
    {
        size_t first_shape_offset = ordered_shapes.size();
        for (int i = start; i < end; ++i)
        {
            const size_t shapes_num = primitive_info[i].primitive_num;
            ordered_shapes.emplace_back(model->indices[shapes_num + 0]);
            ordered_shapes.emplace_back(model->indices[shapes_num + 1]);
            ordered_shapes.emplace_back(model->indices[shapes_num + 2]);
        }
        node->initializeLeaf(first_shape_offset, num_of_primitives, bounds);
        return node;
    }

    AABB centroid_bounds;
    for (int i = start; i < end; ++i)
    {
        centroid_bounds = AABB::merge(centroid_bounds, primitive_info[i].centroid);
    }
    int dim = centroid_bounds.maximumExtent();

    if (centroid_bounds.max[dim] == centroid_bounds.min[dim])
    {
        int first_shape_offset = ordered_shapes.size();
        for (int i = start; i < end; ++i)
        {
            const int shapes_num = primitive_info[i].primitive_num;
            ordered_shapes.emplace_back(model->indices[shapes_num + 0]);
            ordered_shapes.emplace_back(model->indices[shapes_num + 1]);
            ordered_shapes.emplace_back(model->indices[shapes_num + 2]);
        }
        node->initializeLeaf(first_shape_offset, num_of_primitives, bounds);
        return node;
    }
    float pmid = (centroid_bounds.min[dim] + centroid_bounds.max[dim]) / 2.f;
    BVHPrimitiveInfo* mid_ptr = std::partition(&primitive_info[start], &primitive_info[end - 1] + 1,
        [dim, pmid](const BVHPrimitiveInfo& pi)
        {
            return pi.centroid[dim] < pmid;
        });
    int mid = mid_ptr - &primitive_info[0];
    node->initializeInterior(dim, recursiveBuild(primitive_info, start, mid, current_depth + 1, ordered_shapes),
        recursiveBuild(primitive_info, mid, end, current_depth + 1, ordered_shapes));

    return node;
}
