#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Models/RawModel.h"
#include "TextureData.h"
#include "Vertex.h"
#include "Materials/TextureAsset.h"
#include "Materials/MaterialParameters.h"
#include "Objects/Lights/LightCreators/LightCreator.h"
#include "Objects/Lights/LightCreators/DirectionalLightCreator.h"
#include "Objects/Lights/LightCreators/PointLightCreator.h"
#include "Objects/Lights/LightCreators/SpotlightCreator.h"

struct ModelData;
struct MaterialAsset;
class Material;
struct RawModel;

namespace locations
{
    static std::string res_folder_location = "Resources/";
    static std::string textures_folder_location = "Resources/Textures/";
    static std::string image_extension = ".png";
    static std::string models_folder_location = "Resources/Models/";
    static std::string model_extension = ".obj";
    static std::string shader_extension = ".glsl";
    static std::string shader_folder_location = "Resources/Shaders/";
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

    static RawModelAttributes loadSimpleModel(
        const std::vector<float>& positions,
        const std::vector<float>& texture_coords);

    static std::shared_ptr<TextureData> loadImageFromFile(const std::string& file_name);

    static std::shared_ptr<utils::Texture> loadTexture(const std::string& file_name, float lod_value);
    static std::shared_ptr<utils::Texture> loadTexture(const std::string& file_name);

    static std::vector<std::shared_ptr<RawModel>> getAvailableModelAssets();
    static std::shared_ptr<RawModel> findModelAsset(const std::string& file_name);
    static std::vector<std::shared_ptr<MaterialAsset>> getAvailableMaterialAssets();
    static std::shared_ptr<MaterialAsset> findMaterialAsset(const std::string& material_name);
    static std::vector<std::shared_ptr<LightCreator>> getAvailableLightCreators();

private:
    static RawModelAttributes loadModel(const ModelData& model_data);

    static std::shared_ptr<utils::VAO> createVao();
    static void unbindVao();

    static std::shared_ptr<utils::VBO> storeDataInAttributeList(
        unsigned int attribute_number,
        unsigned int coordinate_size,
        const float* data,
        unsigned int count);

    static std::shared_ptr<utils::VBO> bindIndicesBuffer(const unsigned int* indices, unsigned int count);

    inline static std::vector<std::shared_ptr<RawModel>> available_model_assets;
    inline static std::vector<std::shared_ptr<MaterialAsset>> available_material_assets;
    inline static std::vector<std::shared_ptr<TextureAsset>> available_texture_assets;
    inline static std::vector<std::shared_ptr<LightCreator>> AVAILABLE_LIGHT_CREATORS =
    {
            std::make_shared<DirectionalLightCreator>(),
            std::make_shared<PointLightCreator>(),
            std::make_shared<SpotlightCreator>()
    };
};
