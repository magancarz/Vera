#pragma once

#include <memory>
#include <unordered_set>

#include "Voxel.h"
#include "RenderEngine/AccelerationStructures/AABB.h"
#include "OctreeBuildNode.h"
#include "OctreeNode.h"

class MemoryAllocator;

class Octree
{
public:
    explicit Octree(uint32_t max_depth, const std::unordered_set<Voxel>& voxels);

    [[nodiscard]] AABB aabb() const { return octree_aabb; }
    [[nodiscard]] const std::vector<OctreeNode>& nodes() const { return octree_nodes; }

private:
    uint32_t max_depth;

    AABB findOctreeAxisAlignBoundingBox(const std::unordered_set<Voxel>& voxels);

    AABB octree_aabb{};

    std::vector<OctreeNode> createOctree(const std::unordered_set<Voxel>& voxels);
    void insertVoxel(
        std::unique_ptr<OctreeBuildNode>& current_node,
        const Voxel& voxel,
        uint32_t current_depth,
        const glm::vec3& aabb_min,
        const glm::vec3& aabb_max,
        uint32_t& total_nodes);
    std::vector<OctreeNode> flattenOctree(OctreeBuildNode* root_node, uint32_t total_nodes);
    uint32_t flattenNode(std::vector<OctreeNode>& nodes, OctreeBuildNode* octree_build_node, uint32_t* offset);

    std::vector<OctreeNode> octree_nodes;
};
