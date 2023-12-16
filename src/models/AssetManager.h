#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../models/RawModel.h"
#include "TextureData.h"
#include "Vertex.h"
#include "Materials/TextureAsset.h"
#include <Materials/MaterialParameters.h>

struct ModelData;
struct MaterialAsset;
class MaterialFactory;
class MeshMaterial;
class Material;
struct RawModel;

namespace locations
{
    static std::string res_folder_location = "res/";
    static std::string textures_folder_location = "res/textures/";
    static std::string image_extension = ".png";
    static std::string models_folder_location = "res/models/";
    static std::string model_extension = ".obj";
    static std::string shader_extension = ".glsl";
    static std::string shader_folder_location = "res/shaders/";
}

struct RawModelAttributes
{
    std::shared_ptr<utils::VAO> vao;
    std::vector<std::shared_ptr<utils::VBO>> vbos;
    unsigned int indices_size;
};

class AssetManager
{
public:
    static void initializeAssets();

    static std::shared_ptr<RawModel> loadModelAsset(const std::string& file_name);
    static std::shared_ptr<MaterialAsset> createMaterialAsset(const std::string& name, const MaterialParameters& material_parameters = {});
    static std::shared_ptr<RawModel> loadModelFromOBJFileFormat(const std::string& file_name);

    static [[nodiscard]] RawModelAttributes loadRawModel(
        const std::vector<float>& positions,
        const std::vector<float>& texture_coords);

    static std::shared_ptr<TextureData> loadImageFromFile(const std::string& file_name);

    static [[nodiscard]] std::shared_ptr<utils::Texture> createColorTexture(float r, float g, float b);
    static [[nodiscard]] std::shared_ptr<utils::Texture> createColorTexture(uint8_t r, uint8_t g, uint8_t b);
    static [[nodiscard]] std::shared_ptr<utils::Texture> createTexture(int width, int height,
                                                                       const std::vector<char>& data);
    static [[nodiscard]] std::shared_ptr<utils::Texture> loadTexture(const std::string& file_name, float lod_value);
    static [[nodiscard]] std::shared_ptr<utils::Texture> loadTexture(const std::string& file_name);
    static bool loadDataIntoTexture(unsigned int texture_id, const std::vector<char>& data);

    static std::vector<std::shared_ptr<RawModel>> getAvailableModelAssets();
    static std::shared_ptr<RawModel> findModelAsset(const std::string& file_name);
    static std::vector<std::shared_ptr<MaterialAsset>> getAvailableMaterialAssets();
    static std::shared_ptr<MaterialAsset> findMaterialAsset(const std::string& material_name);

private:
    static RawModelAttributes loadModel(const ModelData& model_data);

    static void dealWithAlreadyProcessedVertex(
        const std::shared_ptr<Vertex>& previous_vertex,
        int new_texture_index,
        int new_normal_index,
        std::vector<unsigned int>& indices,
        std::vector<std::shared_ptr<Vertex>>& vertices);

    static void processVertex(
        unsigned int index,
        unsigned int texture_index,
        unsigned int normal_index,
        std::vector<std::shared_ptr<Vertex>>& vertices,
        std::vector<unsigned int>& indices);

    static float convertDataToArrays(
        const std::vector<std::shared_ptr<Vertex>>& vertices,
        const std::vector<glm::vec2>& textures,
        const std::vector<glm::vec3>& normals,
        std::vector<float>& vertices_array,
        std::vector<float>& textures_array,
        std::vector<float>& normals_array);

    static void removeUnusedVertices(const std::vector<std::shared_ptr<Vertex>>& vertices);

    static [[nodiscard]] std::shared_ptr<utils::VAO> createVao();
    static void unbindVao();

    static [[nodiscard]] std::shared_ptr<utils::VBO> storeDataInAttributeList(
        unsigned int attribute_number,
        unsigned int coordinate_size,
        const float* data,
        unsigned int count);

    static [[nodiscard]] std::shared_ptr<utils::VBO> bindIndicesBuffer(const unsigned int* indices, unsigned int count);

    inline static std::vector<std::shared_ptr<RawModel>> available_model_assets;
    inline static std::vector<std::shared_ptr<MaterialAsset>> available_material_assets;
    inline static std::vector<std::shared_ptr<TextureAsset>> available_texture_assets;
};
