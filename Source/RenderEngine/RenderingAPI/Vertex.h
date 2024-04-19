#pragma once

#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <vulkan/vulkan.hpp>

struct Vertex
{
    glm::vec3 position{};
    uint32_t alignment1;
    glm::vec3 normal{};
    uint32_t alignment2;
    glm::vec2 uv{};
    glm::vec2 alignment3{};

    bool operator==(const Vertex& other) const
    {
        return position == other.position && normal == other.normal && uv == other.uv;
    }
};