#pragma once

#include <filesystem>

namespace Assets
{
    inline constexpr const char* DEFAULT_MODEL_NAME{"cube"};
    inline constexpr const char* DEFAULT_MATERIAL_NAME{"white"};

    inline constexpr const char* DEFAULT_DIFFUSE_TEXTURE{"white.png"};
    inline constexpr const char* DEFAULT_NORMAL_MAP{"blue.png"};

    inline static const std::filesystem::path RESOURCES_DIRECTORY_PATH{"Resources"};
    inline static const std::filesystem::path MODELS_DIRECTORY_PATH{"Resources/Models"};
    inline static const std::filesystem::path MATERIALS_DIRECTORY_PATH{"Resources/Materials"};
    inline static const std::filesystem::path TEXTURES_DIRECTORY_PATH{"Resources/Textures"};

    inline constexpr VkFormat DIFFUSE_TEXTURE_FORMAT{VK_FORMAT_R8G8B8A8_SRGB};
    inline constexpr VkFormat NORMAL_MAP_FORMAT{VK_FORMAT_R8G8B8A8_UNORM};
}