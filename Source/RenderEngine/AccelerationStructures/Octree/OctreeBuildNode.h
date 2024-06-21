#pragma once

struct OctreeBuildNode
{
    std::array<std::unique_ptr<OctreeBuildNode>, 8> children;
    AABB aabb;
    bool is_leaf{false};
};