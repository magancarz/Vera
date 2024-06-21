#pragma once
#include "Utils/Algorithms.h"

struct Voxel
{
    float x, y, z;

    bool operator==(const Voxel& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    static constexpr float DEFAULT_VOXEL_SIZE{1.0f / 8.0f};
};

namespace std
{
    template <>
    struct hash<Voxel>
    {
        size_t operator()(Voxel const& voxel) const noexcept
        {
            size_t seed = 0;
            Algorithms::hashCombine(seed, voxel.x, voxel.y, voxel.z);
            return seed;
        }
    };
}
