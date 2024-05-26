#include "OBJModelLoader.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iostream>

#include "Utils/PathBuilder.h"
#include "Utils/Algorithms.h"
#include "RenderEngine/Materials/WavefrontMaterial.h"

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

std::shared_ptr<OBJModel> OBJModelLoader::createFromFile(VulkanFacade& vulkan_facade, AssetManager* asset_manager, const std::string& model_name)
{
    const std::string filepath = PathBuilder(paths::MODELS_DIRECTORY_PATH).append(model_name).build();

    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config))
    {
        std::cerr << "TinyObjReader: " << reader.Error();
        exit(1);
    }

    if (!reader.Warning().empty())
    {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    tinyobj::attrib_t attrib = reader.GetAttrib();
    std::vector<tinyobj::shape_t> shapes = reader.GetShapes();
    std::vector<tinyobj::material_t> obj_materials = reader.GetMaterials();

    for (auto& material : obj_materials)
    {
        MaterialInfo material_info{};
        material_info.brightness = material.emission[0] > 0 || material.emission[1] > 0 || material.emission[2] > 0 ? 10 : -1;
        material_info.fuzziness = material.shininess > 0 ? 0.05 : -1;

        auto texture_name = material.diffuse_texname.empty() ? "white.png" : material.diffuse_texname;
        std::shared_ptr<Texture> texture = asset_manager->fetchTexture(texture_name);
        asset_manager->loadMaterial(
                std::make_shared<WavefrontMaterial>(vulkan_facade, material_info, material.name, std::move(texture)));
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
        for (auto index : shape.mesh.indices)
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
                unique_vertices[vertex] = static_cast<uint32_t>(obj_model_info.vertices.size());
                obj_model_info.vertices.push_back(vertex);
            }
            obj_model_info.indices.push_back(unique_vertices[vertex]);
        }

        obj_model_infos.emplace_back(obj_model_info);
    }

    return std::make_shared<OBJModel>(vulkan_facade, obj_model_infos, model_name);
}