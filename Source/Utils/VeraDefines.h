#pragma once

#include <filesystem>

namespace paths
{
    inline static const std::filesystem::path SHADERS_DIRECTORY_PATH{"Shaders"};
    inline static const std::filesystem::path RESOURCES_DIRECTORY_PATH{"Resources"};
    inline static const std::filesystem::path MODELS_DIRECTORY_PATH{"Resources/Models"};
    inline static const std::filesystem::path MATERIALS_DIRECTORY_PATH{"Resources/Materials"};
    inline static const std::filesystem::path TEXTURES_DIRECTORY_PATH{"Resources/Textures"};
};

namespace constants
{
    inline static const uint32_t DEFAULT_WINDOW_WIDTH{1280};
    inline static const uint32_t DEFAULT_WINDOW_HEIGHT{800};
    inline static const char* DEFAULT_WINDOW_TITLE{"Vera"};
}