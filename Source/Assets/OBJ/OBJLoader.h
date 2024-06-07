#pragma once

#include <Assets/MeshData.h>

#include "Assets/AssetManager.h"
#include "Assets/Model/Model.h"
#include "Assets/Model/ModelData.h"
#include "tiny_obj_loader.h"

class OBJLoader
{
public:
    static MeshData loadAssetsFromFile(const std::string& mesh_name);

    inline static const char* OBJ_FILE_EXTENSION{".obj"};

private:
    static void handleErrorsAndWarnings(const tinyobj::ObjReader& reader);
    static std::vector<MaterialData> loadMaterials(const std::vector<tinyobj::material_t>& obj_materials);
    static std::vector<ModelData> loadModels(
            const tinyobj::ObjReader& reader,
            const std::vector<MaterialData>& obj_materials);
    static Vertex extractVertex(const tinyobj::index_t& index, const tinyobj::attrib_t& attrib);
    static void calculateTangentSpaceVectors(Vertex& first_vertex, Vertex& second_vertex, Vertex& third_vertex);
    static void addVertexToModelInfo(ModelData& model_data, std::unordered_map<Vertex, uint32_t>& unique_vertices, const Vertex& vertex);
};
