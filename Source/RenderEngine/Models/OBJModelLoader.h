#pragma once

#include "Assets/AssetManager.h"
#include "OBJModel.h"
#include "tiny_obj_loader.h"

class OBJModelLoader
{
public:
    static std::unique_ptr<Model> createFromFile(
            MemoryAllocator& memory_allocator,
            AssetManager& asset_manager,
            const std::string& model_name);

private:
    static Vertex extractVertex(const tinyobj::index_t& index, const tinyobj::attrib_t& attrib);
    static void calculateTangentSpaceVectors(Vertex& first_vertex, Vertex& second_vertex, Vertex& third_vertex);
    static void addVertexToModelInfo(OBJModelInfo& obj_model_info, std::unordered_map<Vertex, uint32_t>& unique_vertices, const Vertex& vertex);
};
