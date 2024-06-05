#include "TextureData.h"
#include "Utils/PathBuilder.h"
#include "Assets/Defines.h"

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "Logs/LogSystem.h"

TextureData::TextureData(const std::string& texture_name)
{
    const std::string location = PathBuilder(Assets::TEXTURES_DIRECTORY_PATH).append(texture_name).build();

    data = stbi_load(
            location.c_str(),
            &width,
            &height,
            &number_of_channels,
            expected_number_of_channels);

    if (!data)
    {
        throw std::runtime_error("Failed to load image " + texture_name);
    }

    if (number_of_channels != expected_number_of_channels)
    {
        LogSystem::log(LogSeverity::WARNING, "Encountered different number of channels than expected with texture ", texture_name, "!");
    }
}

TextureData::~TextureData()
{
    stbi_image_free(data);
}
