#pragma once

#include "Assets/AssetManager.h"
#include "Assets/Model/Model.h"
#include "tiny_obj_loader.h"

class OBJLoader
{
public:
    static void loadAssetsFromFile(
            MemoryAllocator& memory_allocator,
            AssetManager& asset_manager,
            const std::string& file_name);

private:
    inline static const char* OBJ_FILE_EXTENSION{".obj"};

    static void handleErrorsAndWarnings(const tinyobj::ObjReader& reader);
    static void loadMaterials(
            AssetManager& asset_manager,
            const std::vector<tinyobj::material_t>& obj_materials);
    static void loadMesh(
            MemoryAllocator& memory_allocator,
            AssetManager& asset_manager,
            const tinyobj::ObjReader& reader,
            const std::vector<tinyobj::material_t>& obj_materials,
            const std::string& mesh_name);
    static Vertex extractVertex(const tinyobj::index_t& index, const tinyobj::attrib_t& attrib);
    static void calculateTangentSpaceVectors(Vertex& first_vertex, Vertex& second_vertex, Vertex& third_vertex);
    static void addVertexToModelInfo(ModelInfo& model_info, std::unordered_map<Vertex, uint32_t>& unique_vertices, const Vertex& vertex);
};
