#include "Octree.h"

#include <chrono>
#include <stack>

#include "glm/ext/quaternion_common.hpp"
#include "Logs/LogSystem.h"

#include "Memory/MemoryAllocator.h"

Octree::Octree(uint32_t max_depth, const std::unordered_set<Voxel>& voxels)
    : max_depth{max_depth}, octree_aabb{findOctreeAxisAlignBoundingBox(voxels)}, nodes{createOctree(voxels)} {}

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
    furthest_point = glm::pow(Voxel::DEFAULT_VOXEL_SIZE, static_cast<float>(next_power));

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

    auto root = std::make_unique<OctreeBuildNode>();
    for (auto& voxel : voxels)
    {
        uint32_t current_depth{0};
        insertVoxel(root, voxel, current_depth, octree_aabb.min, octree_aabb.max);
    }

    std::vector<OctreeNode> final_nodes = flattenOctree(std::move(root));

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
        std::unique_ptr<OctreeBuildNode>& node, const Voxel& voxel, uint32_t current_depth, const glm::vec3& aabb_min, const glm::vec3& aabb_max)
{
    if (!node)
    {
        node = std::make_unique<OctreeBuildNode>();
    }

    if (current_depth == max_depth || aabb_max.x - aabb_min.x <= Voxel::DEFAULT_VOXEL_SIZE)
    {
        node->is_leaf = true;
        return;
    }

    glm::vec3 mid = (aabb_max + aabb_min) / 2.f;
    glm::vec3 new_min{0};
    glm::vec3 new_max{0};

    uint8_t index{0};
    if (voxel.x >= mid.x)
    {
        index |= 1;
        new_min.x = mid.x;
        new_max.x = aabb_max.x;
    }
    else
    {
        new_max.x = mid.x;
        new_min.x = aabb_min.x;
    }

    if (voxel.y >= mid.y)
    {
        index |= 2;
        new_min.y = mid.y;
        new_max.y = aabb_max.y;
    }
    else
    {
        new_max.y = mid.y;
        new_min.y = aabb_min.y;
    }

    if (voxel.z >= mid.z)
    {
        index |= 4;
        new_min.z = mid.z;
        new_max.z = aabb_max.z;
    }
    else
    {
        new_max.z = mid.z;
        new_min.z = aabb_min.z;
    }

    return insertVoxel(node->children[index], voxel, current_depth + 1, new_min, new_max);
}

std::vector<OctreeNode> Octree::flattenOctree(std::unique_ptr<OctreeBuildNode> root_node)
{
    std::vector<OctreeNode> nodes;
    std::stack<OctreeBuildNode*> build_nodes;
    build_nodes.emplace(root_node.get());

    while (!build_nodes.empty())
    {
        OctreeBuildNode* octree_build_node = build_nodes.top();
        build_nodes.pop();

        uint8_t children{0};
        for (size_t i = 0; i < octree_build_node->children.size(); ++i)
        {
            if (octree_build_node->children[i])
            {
                uint8_t temp = 1 << i;
                children |= temp;
                build_nodes.emplace(octree_build_node->children[i].get());
            }
        }
        nodes.emplace_back(children);
    }

    return nodes;
}
