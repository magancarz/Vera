#pragma once

namespace RendererDefines
{
    inline constexpr size_t MAX_NUMBER_OF_LIGHTS{4};
    inline constexpr size_t SHADOW_MAPS_TEXTURES_STARTING_INDEX{0};
    inline constexpr size_t MODEL_TEXTURES_STARTING_INDEX{MAX_NUMBER_OF_LIGHTS};
    inline constexpr size_t G_BUFFER_STARTING_INDEX{SHADOW_MAPS_TEXTURES_STARTING_INDEX};
}