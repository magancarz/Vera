#include "BVHTreeBuilder.h"

#include "BVHTreeNode.h"
#include "RenderEngine/RayTracing/Shapes/ShapeInfo.h"

dmm::DeviceMemoryPointer<BVHTreeTraverser> BVHTreeBuilder::buildAccelerator(
    dmm::DeviceMemoryPointer<Shape*> shapes,
    dmm::DeviceMemoryPointer<ShapeInfo*> shape_infos)
{
    this->shapes = std::move(shapes);
    this->shape_infos = std::move(shape_infos);
    num_of_shapes = this->shapes.size();
    max_shapes_in_node = static_cast<size_t>(std::log(num_of_shapes));
    max_depth = 16;

    std::vector<BVHShapeInfo> primitive_info(num_of_shapes);
    for (size_t i = 0; i < num_of_shapes; ++i)
    {
        primitive_info[i] = BVHShapeInfo{i, this->shape_infos[i]->world_bounds};
    }

    int total_nodes = 0;
    std::vector<Shape*> ordered_shapes;
    const std::shared_ptr<BVHBuildNode> root = recursiveBuild(primitive_info, 0, num_of_shapes, 0, &total_nodes, ordered_shapes);

    this->shapes.copyFrom(ordered_shapes.data());
    nodes = dmm::DeviceMemoryPointer<BVHTreeNode>(total_nodes);
    int offset = 0;
    flattenBVHTree(root, &offset);

    BVHTreeTraverser bvh_tree_traverser{this->shapes.data(), nodes.data()};
    tree_traverser.copyFrom(&bvh_tree_traverser);

    return tree_traverser;
}

std::shared_ptr<BVHBuildNode> BVHTreeBuilder::recursiveBuild(
    std::vector<BVHShapeInfo>& shape_info,
    int start, int end,
    int depth,
    int* total_nodes,
    std::vector<Shape*>& ordered_shapes)
{
    std::shared_ptr<BVHBuildNode> node = std::make_shared<BVHBuildNode>();
    (*total_nodes)++;
    Bounds3f bounds;
    for (int i = start; i < end; ++i)
    {
        bounds = boundsFromUnion(bounds, shape_info[i].bounds);
    }

    const size_t num_of_shapes_in_node = end - start;
    if (num_of_shapes_in_node <= max_shapes_in_node || depth >= max_depth)
    {
        node = createLeafNode(node, start, end, ordered_shapes, shape_info, bounds);
        return node;
    }

    Bounds3f centroid_bounds;
    for (int i = start; i < end; ++i)
    {
        centroid_bounds = boundsFromUnion(centroid_bounds, shape_info[i].centroid);
    }
    int dim = centroid_bounds.maximumExtent();
    int mid = (start + end) / 2;

    if (AdditionalAlgorithms::equal(centroid_bounds.max[dim], centroid_bounds.min[dim]))
    {
        node = createLeafNode(node, start, end, ordered_shapes, shape_info, bounds);
        return node;
    }

    if (num_of_shapes_in_node <= max_shapes_in_node)
    {
        std::nth_element(&shape_info[start], &shape_info[mid], &shape_info[end-1]+1,
            [dim](const BVHShapeInfo& a, const BVHShapeInfo& b)
            {
                return a.centroid[dim] < b.centroid[dim];
            });
    }
    else
    {
        constexpr int num_of_buckets = 12;
        BucketInfo buckets[num_of_buckets];

        for (int i = start; i < end; ++i)
        {
            int b = num_of_buckets * centroid_bounds.offset(shape_info[i].centroid)[dim];
            if (b == num_of_buckets)
            {
                b = num_of_buckets - 1;
            }
            buckets[b].count++;
            buckets[b].bounds = boundsFromUnion(buckets[b].bounds, shape_info[i].bounds);
        }

        float cost[num_of_buckets - 1];
        for (int i = 0; i < num_of_buckets - 1; ++i)
        {
            Bounds3f b0, b1;
            int count0 = 0, count1 = 0;
            for (int j = 0; j <= i; ++j)
            {
                b0 = boundsFromUnion(b0, buckets[j].bounds);
                count0 += buckets[j].count;
            }
            for (int j = i + 1; j < num_of_buckets; ++j)
            {
                b1 = boundsFromUnion(b1, buckets[j].bounds);
                count1 += buckets[j].count;
            }
            cost[i] = 1.f + (count0 * b0.surfaceArea() + count1 * b1.surfaceArea()) / bounds.surfaceArea();
        }

        float min_cost = cost[0];
        int min_cost_split_bucket = 0;
        for (int i = 1; i < num_of_buckets - 1; ++i)
        {
            if (cost[i] < min_cost)
            {
                min_cost = cost[i];
                min_cost_split_bucket = i;
            }
        }

        float leaf_cost = num_of_shapes_in_node;
        if (num_of_shapes_in_node > max_shapes_in_node || min_cost < leaf_cost)
        {
            BVHShapeInfo* pmid = std::partition(&shape_info[start], &shape_info[end - 1] + 1,
                [=](const BVHShapeInfo& shape_info)
                {
                    int b = num_of_buckets * centroid_bounds.offset(shape_info.centroid)[dim];
                    if (b == num_of_buckets)
                    {
                        b = num_of_buckets - 1;
                    }
                    return b <= min_cost_split_bucket;
                });

            mid = pmid - &shape_info[0];
        }
        else
        {
            node = createLeafNode(node, start, end, ordered_shapes, shape_info, bounds);
            return node;
        }
    }

    node->initializeInterior(dim, recursiveBuild(shape_info, start, mid, depth + 1, total_nodes, ordered_shapes),
        recursiveBuild(shape_info, mid, end, depth + 1, total_nodes, ordered_shapes));

    return node;
}

int BVHTreeBuilder::flattenBVHTree(const std::shared_ptr<BVHBuildNode>& node, int* offset)
{
    if (node == nullptr) return *offset;
    BVHTreeNode* linear_node = &nodes[*offset];
    linear_node->bounds = node->bounds;
    const int my_offset = (*offset)++;
    if (node->num_of_shapes > 0)
    {
        linear_node->primitives_offset = node->first_shape_offset;
        linear_node->num_of_shapes = node->num_of_shapes;
    }
    else
    {
        linear_node->split_axis = node->split_axis;
        linear_node->num_of_shapes = 0;
        flattenBVHTree(node->children[0], offset);
        linear_node->second_child_offset = flattenBVHTree(node->children[1], offset);
    }

    return my_offset;
}

std::shared_ptr<BVHBuildNode> BVHTreeBuilder::createLeafNode(
    std::shared_ptr<BVHBuildNode>& node,
    int start, int end,
    std::vector<Shape*>& ordered_shapes,
    const std::vector<BVHShapeInfo>& shape_info,
    const Bounds3f& bounds)
{
    const size_t num_of_shapes_in_node = end - start;
    const size_t first_shape_offset = ordered_shapes.size();
    for (int i = start; i < end; ++i)
    {
        const size_t shapes_num = shape_info[i].shape_index;
        ordered_shapes.push_back(shapes[shapes_num]);
    }
    node->initializeLeaf(first_shape_offset, num_of_shapes_in_node, bounds);
    return node;
}