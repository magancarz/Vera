#pragma once

#include "Device.h"
#include "Vertex.h"

class Model
{
public:
    struct Builder
    {
        std::vector<Vertex> vertices{};
        std::vector<uint32_t> indices{};
    };

    Model(Device& device, const Model::Builder& builder);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void bind(VkCommandBuffer command_buffer);
    void draw(VkCommandBuffer command_buffer);

private:
    void createVertexBuffers(const std::vector<Vertex>& vertices);
    void createIndexBuffers(const std::vector<uint32_t>& indices);

    Device& device;

    VkBuffer vertex_buffer;
    VkDeviceMemory vertex_buffer_memory;
    uint32_t vertex_count;

    bool has_index_buffer{false};
    VkBuffer index_buffer;
    VkDeviceMemory index_buffer_memory;
    uint32_t index_count;
};