#include "AssetManager.h"

#include <iostream>
#include <set>
#include <ranges>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Materials/MaterialAsset.h"
#include "Materials/Material.h"
#include "ModelData.h"
#include "OBJUtils/OBJLoader.h"

void AssetManager::initializeAssets()
{
    loadModelAsset("teapot");
    loadModelAsset("cube");
    loadModelAsset("plane");
    loadModelAsset("sphere");
    loadModelAsset("icosphere");
    loadModelAsset("barrel");
    loadModelAsset("lantern");
    loadModelAsset("cherry");
    loadModelAsset("rock");
    loadModelAsset("monkey");
    loadModelAsset("bunny");

    MaterialParameters white_material_parameters{
        "white", "", "", 0.f, 0.f, 0.f
    };
    createMaterialAsset("white", white_material_parameters);

    MaterialParameters red_material_parameters{
        "red", "", "", 0.f, 0.f, 0.f
    };
    createMaterialAsset("red", red_material_parameters);

    MaterialParameters green_material_parameters{
        "green", "", "", 0.f, 0.f, 0.f
    };
    createMaterialAsset("green", green_material_parameters);

    MaterialParameters blue_material_parameters{
        "blue", "", "", 0.f, 0.f, 0.f
    };
    createMaterialAsset("blue", blue_material_parameters);

    MaterialParameters light_material_parameters{
        "white", "", "specular", 15.f, 0.f, 0.f
    };
    createMaterialAsset("light", light_material_parameters);

    MaterialParameters transparent_material_parameters{
        "white", "", "blue", 0.f, 0.f, 1.5f
    };
    createMaterialAsset("transparent", transparent_material_parameters);

    MaterialParameters mirror_material_parameters{
            "white", "", "red", 0.f, 0.f, 1.5f
    };
    createMaterialAsset("mirror", mirror_material_parameters);

    MaterialParameters barrel_material_parameters{
        "barrel", "barrel_normal", "barrel_specular", 0.f, 1.f, 0.f
    };
    createMaterialAsset("barrel", barrel_material_parameters);

    MaterialParameters lantern_material_parameters{
        "lantern", "", "lantern_specular", 5, 0.f, 0.f
    };
    createMaterialAsset("lantern", lantern_material_parameters);

    MaterialParameters cherry_material_parameters{
        "cherry", "", "", 0.f, 0.8f, 0.f
    };
    createMaterialAsset("cherry", cherry_material_parameters);

    MaterialParameters rock_material_parameters{
        "rock", "rock_normal", "", 0.f, 0.f, 0.f
    };
    createMaterialAsset("rock", rock_material_parameters);
}

std::shared_ptr<RawModel> AssetManager::loadModelAsset(const std::string& file_name)
{
    std::shared_ptr<RawModel> model_asset = findModelAsset(file_name);
    return model_asset ? model_asset : loadModelFromOBJFileFormat(file_name);
}

std::shared_ptr<RawModel> AssetManager::loadModelFromOBJFileFormat(const std::string& file_name)
{
    ModelData model_data = OBJLoader::loadModelDataFromOBJFileFormat(file_name);

    auto raw_model_attributes = loadModel(model_data);
    std::shared_ptr<RawModel> raw_model = std::make_shared<RawModel>(file_name, raw_model_attributes.vao, raw_model_attributes.indices_size, model_data.triangles);
    available_model_assets.push_back(raw_model);
    return raw_model;
}

RawModelAttributes AssetManager::loadModel(const ModelData& model_data)
{
    const std::shared_ptr<utils::VAO> vao = createVao();
    std::shared_ptr<utils::VBO> indices_vbo = bindIndicesBuffer(model_data.indices.data(), model_data.indices.size());
    std::shared_ptr<utils::VBO> positions_vbo = storeDataInAttributeList(0, 3, model_data.positions.data(), model_data.positions.size());
    std::shared_ptr<utils::VBO> texture_coords_vbo = storeDataInAttributeList(1, 2, model_data.texture_coords.data(), model_data.texture_coords.size());
    std::shared_ptr<utils::VBO> normals_vbo = storeDataInAttributeList(2, 3, model_data.normals.data(), model_data.normals.size());
    unbindVao();

    return {vao, {indices_vbo, positions_vbo, texture_coords_vbo, normals_vbo}, static_cast<unsigned int>(model_data.indices.size())};
}

RawModelAttributes AssetManager::loadSimpleModel(
    const std::vector<float>& positions,
    const std::vector<float>& texture_coords)
{
    const std::shared_ptr<utils::VAO> vao = createVao();
    std::shared_ptr<utils::VBO> positions_vbo = storeDataInAttributeList(0, 2, positions.data(), positions.size());
    std::shared_ptr<utils::VBO> texture_coords_vbo = storeDataInAttributeList(
        1, 2, texture_coords.data(), texture_coords.size());
    unbindVao();

    return {vao, {positions_vbo, texture_coords_vbo}, static_cast<unsigned int>(positions.size())};
}

std::shared_ptr<utils::Texture> AssetManager::loadTexture(const std::string& file_name, const float lod_value)
{
    const auto texture = std::make_shared<utils::Texture>();
    glBindTexture(GL_TEXTURE_2D, texture->texture_id);

    const auto data = loadImageFromFile(file_name);
    const int width = data->width,
              height = data->height;

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data->texture_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, lod_value);

    // check if anisotropic filtering is supported
    int no_of_extensions = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &no_of_extensions);

    std::set<std::string> ogl_extensions;
    for (int i = 0; i < no_of_extensions; i++)
    {
        ogl_extensions.insert(reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i)));
    }

    if (ogl_extensions.contains("GL_EXT_texture_filter_anisotropic"))
    {
        float max_aniso = 0.0f;
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_aniso);
        const float amount = std::min(4.0f, max_aniso);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, amount);
    }
    else
    {
        std::cout << "WARNING: Anisotropic filtering is not supported\n";
    }

    stbi_image_free(data->texture_data);

    return texture;
}

std::shared_ptr<TextureData> AssetManager::loadImageFromFile(const std::string& file_name)
{
    const std::string file = locations::textures_folder_location + file_name + locations::image_extension;
    int width, height, format;
    unsigned char* image = stbi_load(file.c_str(), &width, &height, &format, STBI_rgb_alpha);
    if (image == nullptr)
        std::cout << "Failed to load " + file_name + " image!\n";

    return std::make_shared<TextureData>(image, width, height);
}

std::shared_ptr<utils::Texture> AssetManager::loadTexture(const std::string& file_name)
{
    return loadTexture(file_name, -0.4f);
}

std::shared_ptr<utils::VAO> AssetManager::createVao()
{
    auto vao = std::make_shared<utils::VAO>();
    glBindVertexArray(vao->vao_id);

    return vao;
}

void AssetManager::unbindVao()
{
    glBindVertexArray(0);
}

std::shared_ptr<utils::VBO> AssetManager::storeDataInAttributeList(
    const unsigned int attribute_number,
    const unsigned int coordinate_size,
    const float* data,
    const unsigned int count)
{
    const auto vbo = std::make_shared<utils::VBO>();
    glBindBuffer(GL_ARRAY_BUFFER, vbo->vbo_id);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(float), data, GL_STATIC_DRAW);
    glVertexAttribPointer(attribute_number, coordinate_size, GL_FLOAT, false, 0, nullptr);
    return vbo;
}

std::shared_ptr<utils::VBO> AssetManager::bindIndicesBuffer(const unsigned int* indices, const unsigned int count)
{
    const auto vbo = std::make_shared<utils::VBO>();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo->vbo_id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(unsigned int), indices, GL_STATIC_DRAW);
    return vbo;
}

std::vector<std::shared_ptr<RawModel>> AssetManager::getAvailableModelAssets()
{
    return available_model_assets;
}

std::shared_ptr<RawModel> AssetManager::findModelAsset(const std::string& file_name)
{
    const auto found_model = std::find_if(available_model_assets.begin(), available_model_assets.end(), [&](const std::shared_ptr<RawModel>& model_asset)
    {
        return model_asset->model_name == file_name;
    });

    return found_model != available_model_assets.end()? *found_model : nullptr;
}

std::vector<std::shared_ptr<MaterialAsset>> AssetManager::getAvailableMaterialAssets()
{
    return available_material_assets;
}

std::shared_ptr<MaterialAsset> AssetManager::findMaterialAsset(const std::string& material_name)
{
    const auto found_material = std::find_if(available_material_assets.begin(), available_material_assets.end(), [&](const std::shared_ptr<MaterialAsset>& material_asset)
    {
        return material_asset->material->name == material_name;
    });

    return found_material != available_material_assets.end()? *found_material : nullptr;
}

std::shared_ptr<MaterialAsset> AssetManager::createMaterialAsset(const std::string& name, const MaterialParameters& material_parameters)
{
    const auto color_texture = std::make_shared<TextureAsset>(loadTexture(material_parameters.color_texture_name));
    const auto normal_map_texture = material_parameters.normal_map_texture_name.length() != 0 ? std::make_shared<TextureAsset>(loadTexture(material_parameters.normal_map_texture_name)) : nullptr;
    const auto specular_map_texture = material_parameters.specular_map_texture_name.length() != 0 ? std::make_shared<TextureAsset>(loadTexture(material_parameters.specular_map_texture_name)) : nullptr;
    auto material = std::make_shared<Material>(name, color_texture, normal_map_texture, specular_map_texture, material_parameters);
    dmm::DeviceMemoryPointer<Material> dmm_material{};
    dmm_material.copyFrom(material.get());
    auto material_asset = std::make_shared<MaterialAsset>(dmm_material, material, color_texture, normal_map_texture, specular_map_texture);
	available_material_assets.push_back(material_asset);
    return material_asset;
}