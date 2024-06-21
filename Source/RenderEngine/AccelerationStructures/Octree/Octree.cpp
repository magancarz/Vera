#include "Octree.h"

#include <bitset>
#include <chrono>
#include <queue>
#include <stack>

#include "glm/ext/quaternion_common.hpp"
#include "Logs/LogSystem.h"

#include "Memory/MemoryAllocator.h"

Octree::Octree(uint32_t max_depth, const std::unordered_set<Voxel>& voxels)
    : max_depth{max_depth}, octree_aabb{findOctreeAxisAlignBoundingBox(voxels)}, octree_nodes{createOctree(voxels)} {}

AABB Octree::findOctreeAxisAlignBoundingBox(const std::unordered_set<Voxel>& voxels)
{
    float furthest_point{0};
    for (auto& voxel : voxels)
    {
        furthest_point = glm::max(furthest_point, voxel.x);
        furthest_point = glm::max(furthest_point, voxel.y);
        furthest_point = glm::max(furthest_point, voxel.z);
    }

    float log_result = std::log(furthest_point) / std::log(Voxel::DEFAULT_VOXEL_SIZE);
    int next_power = static_cast<int>(std::ceil(log_result));
    furthest_point = glm::pow(Voxel::DEFAULT_VOXEL_SIZE, static_cast<float>(next_power)) * 2;

    AABB aabb
    {
        .min = glm::vec3{-furthest_point, -furthest_point, -furthest_point},
        .max = glm::vec3{furthest_point, furthest_point, furthest_point}
    };

    return aabb;
}

std::vector<OctreeNode> Octree::createOctree(const std::unordered_set<Voxel>& voxels)
{
    auto start = std::chrono::high_resolution_clock::now();

    uint32_t total_nodes{0};
    std::unique_ptr<OctreeBuildNode> root;
    for (auto& voxel : voxels)
    {
        uint32_t current_depth{0};
        insertVoxel(root, voxel, current_depth, octree_aabb.min, octree_aabb.max, total_nodes);
    }

    std::vector<OctreeNode> final_nodes = flattenOctree(root.get(), total_nodes);

    auto end = std::chrono::high_resolution_clock::now();
    LogSystem::log(
        LogSeverity::LOG,
        "Created octree from ",
        voxels.size(),
        " voxels in ",
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(),
        " millis.");

    return final_nodes;
}

void Octree::insertVoxel(
        std::unique_ptr<OctreeBuildNode>& current_node,
        const Voxel& voxel,
        uint32_t current_depth,
        const glm::vec3& aabb_min,
        const glm::vec3& aabb_max,
        uint32_t& total_nodes)
{
    if (!current_node)
    {
        ++total_nodes;
        current_node = std::make_unique<OctreeBuildNode>();
        current_node->aabb = AABB{aabb_min, aabb_max};
    }

    if (current_depth == max_depth || aabb_max.x - aabb_min.x <= Voxel::DEFAULT_VOXEL_SIZE)
    {
        current_node->is_leaf = true;
        return;
    }

    glm::vec3 mid = (aabb_max + aabb_min) / 2.f;
    glm::vec3 new_min{0};
    glm::vec3 new_max{0};

    uint8_t index{0};
    if (voxel.x >= mid.x)
    {
        new_min.x = mid.x;
        new_max.x = aabb_max.x;
    }
    else
    {
        index |= 4;
        new_max.x = mid.x;
        new_min.x = aabb_min.x;
    }

    if (voxel.y >= mid.y)
    {
        new_min.y = mid.y;
        new_max.y = aabb_max.y;
    }
    else
    {
        index |= 2;
        new_max.y = mid.y;
        new_min.y = aabb_min.y;
    }

    if (voxel.z >= mid.z)
    {
        new_min.z = mid.z;
        new_max.z = aabb_max.z;
    }
    else
    {
        index |= 1;
        new_max.z = mid.z;
        new_min.z = aabb_min.z;
    }

    return insertVoxel(current_node->children[index], voxel, current_depth + 1, new_min, new_max, total_nodes);
}

std::vector<OctreeNode> Octree::flattenOctree(OctreeBuildNode* root_node, uint32_t total_nodes)
{
    std::vector<OctreeNode> nodes(total_nodes);
    uint32_t offset{0};
    flattenNode(nodes, root_node, &offset);

    return nodes;
}

uint32_t Octree::flattenNode(std::vector<OctreeNode>& nodes, OctreeBuildNode* octree_build_node, uint32_t* offset)
{
    if (!octree_build_node)
    {
        return 0;
    }
    const uint32_t my_offset = (*offset)++;
    OctreeNode& node = nodes[my_offset];

    glm::vec4 aabb_min{octree_build_node->aabb.min, 0.0};
    glm::vec4 aabb_max{octree_build_node->aabb.max, 0.0};

    node.children_and_color = glm::vec4{0};
    node.aabb_min = aabb_min;
    node.aabb_max = aabb_max;

    if (octree_build_node->is_leaf)
    {
        return my_offset;
    }

    uint8_t children{0};
    for (size_t i = 0; i < octree_build_node->children.size(); ++i)
    {
        if (octree_build_node->children[i])
        {
            uint8_t temp = 1 << i;
            children |= temp;
            node.child_offsets[i] = flattenNode(nodes, octree_build_node->children[i].get(), offset);
        }
    }

    uint32_t children_and_color{0};
    children_and_color |= children << 24;
    node.children_and_color.x = static_cast<float>(children_and_color);

    return my_offset;
}
