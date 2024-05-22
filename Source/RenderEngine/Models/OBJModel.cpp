#include "OBJModel.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <unordered_map>
#include <iostream>

#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "Utils/Algorithms.h"
#include "RenderEngine/Models/RayTracingAccelerationStructureBuilder.h"
#include "Utils/VeraDefines.h"

OBJModel::OBJModel(VulkanFacade& device, const OBJModel::Builder& builder, std::string model_name)
    : Model(std::move(model_name)), device{device}
{
    createVertexBuffers(builder.vertices);
    createIndexBuffers(builder.indices);
}

void OBJModel::createVertexBuffers(const std::vector<Vertex>& vertices)
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
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    device.copyBuffer(staging_buffer.getBuffer(), vertex_buffer->getBuffer(), buffer_size);
}

void OBJModel::createIndexBuffers(const std::vector<uint32_t>& indices)
{
    index_count = static_cast<uint32_t>(indices.size());
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
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    device.copyBuffer(staging_buffer.getBuffer(), index_buffer->getBuffer(), buffer_size);
}

namespace std
{
    template <>
    struct hash<Vertex> {
        size_t operator()(Vertex const &vertex) const
        {
            size_t seed = 0;
            algorithms::hashCombine(seed, vertex.position, vertex.normal, vertex.uv);
            return seed;
        }
    };
}

std::unique_ptr<Model> OBJModel::createModelFromFile(VulkanFacade& device, const std::string& model_name)
{
    const std::string filepath = (paths::MODELS_DIRECTORY_PATH / model_name).generic_string() + ".obj";

    Builder builder{};
    builder.loadModel(filepath);
    return std::make_unique<OBJModel>(device, builder, model_name);
}

void OBJModel::Builder::loadModel(const std::string& filepath)
{
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    tinyobj::attrib_t attrib = reader.GetAttrib();
    std::vector<tinyobj::shape_t> shapes = reader.GetShapes();
    std::vector<tinyobj::material_t> obj_materials = reader.GetMaterials();

    vertices.clear();
    indices.clear();

    std::unordered_map<Vertex, uint32_t> unique_vertices{};
    for (const auto& shape : shapes)
    {
        for (int i = 0; i < shape.mesh.indices.size(); ++i)
        {
            const auto index = shape.mesh.indices[i];
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

    for (size_t i = 0; i < indices.size(); i += 3)
    {
        Vertex first_vertex = vertices[indices[i + 0]];
        Vertex second_vertex = vertices[indices[i + 1]];
        Vertex third_vertex = vertices[indices[i + 2]];
    }
}
