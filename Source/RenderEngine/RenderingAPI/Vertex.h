#pragma once

#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <vulkan/vulkan.hpp>

struct Vertex
{
    glm::vec3 position{};
    uint32_t material_index{0};
    glm::vec3 normal{};
    unsigned int alignment1;
    glm::vec2 uv{};
    glm::vec2 alignment2{};

    bool operator==(const Vertex& other) const
    {
        return position == other.position && normal == other.normal && uv == other.uv && material_index == other.material_index;
    }
};