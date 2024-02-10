#include "Model.h"

#include "VulkanDefines.h"

Model::Model(Device& device, const Model::Builder& builder)
    : device{device}
{
    createVertexBuffers(builder.vertices);
    createIndexBuffers(builder.indices);
}

void Model::createVertexBuffers(const std::vector<Vertex>& vertices)
{
    vertex_count = static_cast<uint32_t>(vertices.size());
    assert(vertex_count >= 3 && "Vertex count must be at least 3.");
    VkDeviceSize buffer_size = sizeof(vertices[0]) * vertex_count;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    device.createBuffer(
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

    void* data;
    vkMapMemory(device.getDevice(), staging_buffer_memory, 0, buffer_size, 0, &data);
    memcpy(data, vertices.data(), static_cast<size_t>(buffer_size));
    vkUnmapMemory(device.getDevice(), staging_buffer_memory);

    device.createBuffer(
            buffer_size,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vertex_buffer,
            vertex_buffer_memory);

    device.copyBuffer(staging_buffer, vertex_buffer, buffer_size);

    vkDestroyBuffer(device.getDevice(), staging_buffer, nullptr);
    vkFreeMemory(device.getDevice(), staging_buffer_memory, nullptr);
}

void Model::createIndexBuffers(const std::vector<uint32_t>& indices)
{
    index_count = static_cast<uint32_t>(indices.size());
    has_index_buffer = index_count > 0;

    if (!has_index_buffer)
    {
        return;
    }

    VkDeviceSize buffer_size = sizeof(indices[0]) * index_count;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    device.createBuffer(
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

    void *data;
    vkMapMemory(device.getDevice(), staging_buffer_memory, 0, buffer_size, 0, &data);
    memcpy(data, indices.data(), static_cast<size_t>(buffer_size));
    vkUnmapMemory(device.getDevice(), staging_buffer_memory);

    device.createBuffer(
            buffer_size,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            index_buffer,
            index_buffer_memory);

    device.copyBuffer(staging_buffer, index_buffer, buffer_size);

    vkDestroyBuffer(device.getDevice(), staging_buffer, nullptr);
    vkFreeMemory(device.getDevice(), staging_buffer_memory, nullptr);
}

Model::~Model()
{
    vkDestroyBuffer(device.getDevice(), vertex_buffer, VulkanDefines::NO_CALLBACK);
    vkFreeMemory(device.getDevice(), vertex_buffer_memory, VulkanDefines::NO_CALLBACK);

    if (has_index_buffer)
    {
        vkDestroyBuffer(device.getDevice(), index_buffer, VulkanDefines::NO_CALLBACK);
        vkFreeMemory(device.getDevice(), index_buffer_memory, VulkanDefines::NO_CALLBACK);
    }
}

void Model::bind(VkCommandBuffer command_buffer)
{
    VkBuffer buffers[] = {vertex_buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, buffers, offsets);

    if (has_index_buffer)
    {
        vkCmdBindIndexBuffer(command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
    }
}

void Model::draw(VkCommandBuffer command_buffer)
{
    if (has_index_buffer)
    {
        vkCmdDrawIndexed(command_buffer, index_count, 1, 0, 0, 0);
    }
    else
    {
        vkCmdDraw(command_buffer, vertex_count, 1, 0, 0);
    }
}