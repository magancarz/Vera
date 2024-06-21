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

struct Mesh;

class VoxelUtils
{
public:
    static std::unordered_set<Voxel> voxelize(const Mesh& mesh);
};
