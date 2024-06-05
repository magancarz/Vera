#include "VeraMaterial.h"

#include <fstream>

#include "Assets/Defines.h"
#include "Logs/LogSystem.h"
#include "Utils/PathBuilder.h"

void VeraMaterial::loadAssetFromFile(AssetManager& asset_manager, const std::string& asset_name)
{
    LogSystem::log(LogSeverity::LOG, "Trying to load material named ", asset_name.c_str(), "...");

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

        MaterialInfo material_info{};
        std::string color_x, color_y, color_z;
        iss >> color_x;
        iss >> color_y;
        iss >> color_z;

        std::string value;
        iss >> value; // brightness
        iss >> value; // fuzziness
        iss >> value; // refractive index

        std::string texture_name;
        iss >> texture_name;

        file_stream.close();

        material_info.diffuse_texture = asset_manager.fetchTexture(texture_name, VK_FORMAT_R8G8B8A8_SRGB);
        material_info.normal_texture = asset_manager.fetchTexture("blue.png", VK_FORMAT_R8G8B8A8_UNORM);

        LogSystem::log(LogSeverity::LOG, "Loading material from file ended in success");
        asset_manager.storeMaterial(std::make_unique<Material>(material_info));
    }

    LogSystem::log(LogSeverity::ERROR, "Failed to load material");
}
