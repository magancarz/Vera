#pragma once

#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <vulkan/vulkan.hpp>

struct Vertex
{
    glm::vec2 position;
    glm::vec3 color;

    static std::vector<VkVertexInputBindingDescription> getBindingDescriptions()
    {
        return {{.binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX}};
    }

    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions()
    {
        std::vector<VkVertexInputAttributeDescription> attribute_description(2);
        attribute_description[0].binding = 0;
        attribute_description[0].location = 0;
        attribute_description[0].format = VK_FORMAT_R32G32_SFLOAT;
        attribute_description[0].offset = offsetof(Vertex, position);

        attribute_description[1].binding = 0;
        attribute_description[1].location = 1;
        attribute_description[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_description[1].offset = offsetof(Vertex, color);

        return attribute_description;
    }
};