#pragma once

#include <filesystem>

namespace Assets
{
    inline constexpr const char* DEFAULT_MODEL_NAME{"cube"};
    inline constexpr const char* DEFAULT_MATERIAL_NAME{"white"};

    inline static const std::filesystem::path RESOURCES_DIRECTORY_PATH{"Resources"};
    inline static const std::filesystem::path MODELS_DIRECTORY_PATH{"Resources/Models"};
    inline static const std::filesystem::path MATERIALS_DIRECTORY_PATH{"Resources/Materials"};
    inline static const std::filesystem::path TEXTURES_DIRECTORY_PATH{"Resources/Textures"};
}