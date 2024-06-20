#pragma once

#include <unordered_set>
#include <vector>
#include <chrono>

#include "Voxel.h"
#include "Assets/Model/Model.h"
#include "Assets/Model/Vertex.h"
#include "Logs/LogSystem.h"
#include "Memory/MemoryAllocator.h"
#include "Utils/Algorithms.h"

namespace std
{
    template <>
    struct hash<Voxel> {
        size_t operator()(Voxel const& voxel) const noexcept
        {
            size_t seed = 0;
            Algorithms::hashCombine(seed, voxel.x, voxel.y, voxel.z);
            return seed;
        }
    };
}

class VoxelUtils
{
public:
    static std::unordered_set<Voxel> voxelize(const Model& model);
};
