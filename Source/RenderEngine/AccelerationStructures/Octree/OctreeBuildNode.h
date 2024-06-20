#pragma once

struct OctreeBuildNode
{
    std::array<std::unique_ptr<OctreeBuildNode>, 8> children;
    bool is_leaf{false};
};