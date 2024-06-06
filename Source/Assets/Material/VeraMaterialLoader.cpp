#include "VeraMaterialLoader.h"

#include <fstream>

#include "Assets/Defines.h"
#include "Logs/LogSystem.h"
#include "Utils/PathBuilder.h"

void VeraMaterialLoader::loadAssetFromFile(AssetManager& asset_manager, const std::string& asset_name)
{
    LogSystem::log(LogSeverity::LOG, "Trying to load material ", asset_name.c_str(), "...");

    const std::string filepath = PathBuilder(Assets::MATERIALS_DIRECTORY_PATH).append(asset_name).fileExtension(VERA_MATERIAL_FILE_EXTENSION).build();
    std::ifstream file_stream(filepath);
    std::string input;

    if (file_stream.is_open() && getline(file_stream, input))
    {
        std::stringstream iss(input);
        std::string material_name;
        iss >> material_name;

        std::string type;
        iss >> type;

        std::string color_x;
        iss >> color_x;

        std::string color_y;
        iss >> color_y;

        std::string color_z;
        iss >> color_z;

        std::string value;
        iss >> value; // brightness
        iss >> value; // fuzziness
        iss >> value; // refractive index

        std::string texture_name;
        iss >> texture_name;

        file_stream.close();

        MaterialInfo material_info{};
        material_info.name = material_name;
        material_info.diffuse_texture = asset_manager.fetchDiffuseTexture(texture_name);
        material_info.normal_texture = asset_manager.fetchNormalMap(Assets::DEFAULT_NORMAL_MAP);

        LogSystem::log(LogSeverity::LOG, "Loading material ", asset_name.c_str(), " ended in success");
        asset_manager.storeMaterial(std::make_unique<Material>(material_info));
    }

    LogSystem::log(LogSeverity::ERROR, "Failed to load material ", asset_name.c_str());
}
