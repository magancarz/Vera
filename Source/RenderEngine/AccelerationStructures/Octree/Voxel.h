#pragma once

struct Voxel
{
    float x, y, z;
    bool operator==(const Voxel& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    static constexpr float DEFAULT_VOXEL_SIZE{1.0f / 8.0f};
};
