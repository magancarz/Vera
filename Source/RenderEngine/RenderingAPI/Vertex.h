#pragma once

#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <vulkan/vulkan.hpp>

struct Vertex
{
    glm::vec2 position;

    static std::vector<VkVertexInputBindingDescription> getBindingDescriptions()
    {
        return {{.binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX}};
    }

    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions()
    {
        return {{.location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = 0}};
    }
};