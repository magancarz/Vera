#pragma once

#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <vulkan/vulkan.hpp>

struct Vertex
{
    alignas(16) glm::vec3 position{};
    alignas(16) glm::vec3 normal{};
    alignas(16) glm::vec2 uv{};
    alignas(16) glm::vec3 tangent{};

    static std::vector<VkVertexInputBindingDescription> getBindingDescriptions()
    {
        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return {binding};
    }

    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions()
    {
        std::vector<VkVertexInputAttributeDescription> attribute_description{};
        attribute_description.push_back({.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = static_cast<uint32_t>(offsetof(Vertex, position))});
        attribute_description.push_back({.location = 1, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = static_cast<uint32_t>(offsetof(Vertex, uv))});

        return attribute_description;
    }

    bool operator==(const Vertex& other) const
    {
        return position == other.position && normal == other.normal && uv == other.uv;
    }
};