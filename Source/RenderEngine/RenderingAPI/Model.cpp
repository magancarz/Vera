#include "Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <unordered_map>

#include "VulkanDefines.h"
#include "Utils/Algorithms.h"

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
    uint32_t vertex_size = sizeof(vertices[0]);

    Buffer staging_buffer
    {
        device,
        vertex_size,
        vertex_count,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    };

    staging_buffer.map();
    staging_buffer.writeToBuffer((void*)vertices.data());

    vertex_buffer = std::make_unique<Buffer>
    (
        device,
        vertex_size,
        vertex_count,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    device.copyBuffer(staging_buffer.getBuffer(), vertex_buffer->getBuffer(), buffer_size);
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
    uint32_t index_size = sizeof(indices[0]);

    Buffer staging_buffer
    {
        device,
        index_size,
        index_count,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    };

    staging_buffer.map();
    staging_buffer.writeToBuffer((void*)indices.data());

    index_buffer = std::make_unique<Buffer>
    (
        device,
        index_size,
        index_count,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    device.copyBuffer(staging_buffer.getBuffer(), index_buffer->getBuffer(), buffer_size);
}

void Model::bind(VkCommandBuffer command_buffer)
{
    VkBuffer buffers[] = {vertex_buffer->getBuffer()};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, buffers, offsets);

    if (has_index_buffer)
    {
        vkCmdBindIndexBuffer(command_buffer, index_buffer->getBuffer(), 0, VK_INDEX_TYPE_UINT32);
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

namespace std
{
    template <>
    struct hash<Vertex> {
        size_t operator()(Vertex const &vertex) const
        {
            size_t seed = 0;
            Algorithms::hashCombine(seed, vertex.position, vertex.normal, vertex.uv);
            return seed;
        }
    };
}

std::unique_ptr<Model> Model::createModelFromFile(Device& device, const std::string& filepath)
{
    Builder builder{};
    builder.loadModel(filepath);
    return std::make_unique<Model>(device, builder);
}

void Model::Builder::loadModel(const std::string& filepath)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str()))
    {
        throw std::runtime_error(warn + err);
    }

    vertices.clear();
    indices.clear();

    std::unordered_map<Vertex, uint32_t> unique_vertices{};
    for (const auto& shape : shapes)
    {
        for (const auto& index : shape.mesh.indices)
        {
            Vertex vertex{};

            if (index.vertex_index >= 0)
            {
                vertex.position =
                {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2],
                };
            }

            if (index.normal_index >= 0)
            {
                vertex.normal =
                {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2],
                };
            }

            if (index.texcoord_index >= 0)
            {
                vertex.uv =
                {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    attrib.texcoords[2 * index.texcoord_index + 1],
                };
            }

            if (unique_vertices.count(vertex) == 0)
            {
                unique_vertices[vertex] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }
            indices.push_back(unique_vertices[vertex]);
        }
    }
}