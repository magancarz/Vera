#pragma once

#include "Device.h"
#include "Vertex.h"

class Model
{
public:
    Model(Device& device, const std::vector<Vertex>& vertices);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void bind(VkCommandBuffer command_buffer);
    void draw(VkCommandBuffer command_buffer);

private:
    void createVertexBuffers(const std::vector<Vertex>& vertices);

    Device& device;
    VkBuffer vertex_buffer;
    VkDeviceMemory vertex_buffer_memory;
    uint32_t vertex_count;
};