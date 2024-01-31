#include "Model.h"

#include "VulkanDefines.h"

Model::Model(Device& device, const std::vector<Vertex>& vertices)
    : device{device}
{
    createVertexBuffers(vertices);
}

void Model::createVertexBuffers(const std::vector<Vertex>& vertices)
{
    vertex_count = static_cast<uint32_t>(vertices.size());
    assert(vertex_count >= 3 && "Vertex count must be at least 3.");
    VkDeviceSize buffer_size = sizeof(vertices[0]) * vertex_count;
    device.createBuffer(
            buffer_size,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertex_buffer,
            vertex_buffer_memory);

    void* data;
    vkMapMemory(device.getDevice(), vertex_buffer_memory, 0, buffer_size, 0, &data);
    memcpy(data, vertices.data(), static_cast<size_t>(buffer_size));
    vkUnmapMemory(device.getDevice(), vertex_buffer_memory);
}

Model::~Model()
{
    vkDestroyBuffer(device.getDevice(), vertex_buffer, VulkanDefines::NO_CALLBACK);
    vkFreeMemory(device.getDevice(), vertex_buffer_memory, VulkanDefines::NO_CALLBACK);
}

void Model::bind(VkCommandBuffer command_buffer)
{
    VkBuffer buffers[] = {vertex_buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, buffers, offsets);
}

void Model::draw(VkCommandBuffer command_buffer)
{
    vkCmdDraw(command_buffer, vertex_count, 1, 0, 0);
}