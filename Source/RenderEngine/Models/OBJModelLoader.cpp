#include "OBJModelLoader.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iostream>

#include "Logs/LogSystem.h"
#include "Utils/PathBuilder.h"
#include "Utils/Algorithms.h"
#include "RenderEngine/Materials/WavefrontMaterial.h"

namespace std
{
    template <>
    struct hash<Vertex> {
        size_t operator()(Vertex const &vertex) const noexcept
        {
            size_t seed = 0;
            algorithms::hashCombine(seed, vertex.position, vertex.normal, vertex.uv);
            return seed;
        }
    };
}

std::unique_ptr<Model> OBJModelLoader::createFromFile(MemoryAllocator& memory_allocator, AssetManager& asset_manager, const std::string& model_name)
{
    const std::string filepath = PathBuilder(paths::MODELS_DIRECTORY_PATH).append(model_name).build();

    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config))
    {
        LogSystem::log(LogSeverity::FATAL, "TinyObjReader: ", reader.Error());
        throw std::runtime_error("TinyObjReader: " + reader.Error());
    }

    if (!reader.Warning().empty())
    {
        LogSystem::log(LogSeverity::WARNING, "TinyObjReader: ", reader.Warning());
    }

    tinyobj::attrib_t attrib = reader.GetAttrib();
    std::vector<tinyobj::shape_t> shapes = reader.GetShapes();
    std::vector<tinyobj::material_t> obj_materials = reader.GetMaterials();

    for (auto& material : obj_materials)
    {
        MaterialInfo material_info{};
        material_info.brightness = material.emission[0] > 0 || material.emission[1] > 0 || material.emission[2] > 0 ? 10 : -1;
        material_info.fuzziness = material.shininess > 0 ? 0.05 : -1;

        auto diffuse_texture_name = material.diffuse_texname.empty() ? "white.png" : material.diffuse_texname;
        Texture* diffuse_texture = asset_manager.fetchTexture(diffuse_texture_name, VK_FORMAT_R8G8B8A8_SRGB);

        auto normal_texture_name = material.displacement_texname.empty() ? "blue.png" : material.displacement_texname;
        Texture* normal_texture = asset_manager.fetchTexture(normal_texture_name, VK_FORMAT_R8G8B8A8_UNORM);

        asset_manager.storeMaterial(
                std::make_unique<WavefrontMaterial>(
                        memory_allocator,
                        material_info,
                        material.name,
                        std::move(diffuse_texture),
                        std::move(normal_texture)));
    }

    std::vector<OBJModelInfo> obj_model_infos;
    obj_model_infos.reserve(shapes.size());
    for (const auto& shape : shapes)
    {
        OBJModelInfo obj_model_info{};
        obj_model_info.name = shape.name;
        int material_id = shape.mesh.material_ids[0];
        obj_model_info.required_material = material_id >= 0 && !obj_materials.empty() ? obj_materials[material_id].name : "white";

        std::unordered_map<Vertex, uint32_t> unique_vertices{};
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3)
        {
            Vertex first_vertex = extractVertex(shape.mesh.indices[i + 0], attrib);
            Vertex second_vertex = extractVertex(shape.mesh.indices[i + 1], attrib);
            Vertex third_vertex = extractVertex(shape.mesh.indices[i + 2], attrib);

            calculateTangentSpaceVectors(first_vertex, second_vertex, third_vertex);

            addVertexToModelInfo(obj_model_info, unique_vertices, first_vertex);
            addVertexToModelInfo(obj_model_info, unique_vertices, second_vertex);
            addVertexToModelInfo(obj_model_info, unique_vertices, third_vertex);
        }

        obj_model_infos.emplace_back(obj_model_info);
    }

    return std::make_unique<OBJModel>(memory_allocator, obj_model_infos, model_name);
}

Vertex OBJModelLoader::extractVertex(const tinyobj::index_t& index, const tinyobj::attrib_t& attrib)
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
                1.0 - attrib.texcoords[2 * index.texcoord_index + 1],
        };
    }

    return vertex;
}

void OBJModelLoader::calculateTangentSpaceVectors(Vertex& first_vertex, Vertex& second_vertex, Vertex& third_vertex)
{
    glm::vec3 edge1 = second_vertex.position - first_vertex.position;
    glm::vec3 edge2 = third_vertex.position - first_vertex.position;
    glm::vec2 delta_UV1 = second_vertex.uv - first_vertex.uv;
    glm::vec2 delta_UV2 = third_vertex.uv - first_vertex.uv;

    glm::vec3 tangent;
    float f = 1.0f / (delta_UV1.x * delta_UV2.y - delta_UV2.x * delta_UV1.y);
    tangent.x = f * (delta_UV2.y * edge1.x - delta_UV1.y * edge2.x);
    tangent.y = f * (delta_UV2.y * edge1.y - delta_UV1.y * edge2.y);
    tangent.z = f * (delta_UV2.y * edge1.z - delta_UV1.y * edge2.z);
    first_vertex.tangent = second_vertex.tangent = third_vertex.tangent = normalize(tangent);
}

void OBJModelLoader::addVertexToModelInfo(OBJModelInfo& obj_model_info, std::unordered_map<Vertex, uint32_t>& unique_vertices, const Vertex& vertex)
{
    if (!unique_vertices.contains(vertex))
    {
        unique_vertices[vertex] = static_cast<uint32_t>(obj_model_info.vertices.size());
        obj_model_info.vertices.push_back(vertex);
    }
    obj_model_info.indices.push_back(unique_vertices[vertex]);
}