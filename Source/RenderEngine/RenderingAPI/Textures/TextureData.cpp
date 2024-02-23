#include "TextureData.h"

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

TextureData::TextureData(const std::string& filepath)
{
    data = stbi_load(
            filepath.c_str(),
            &width,
            &height,
            &number_of_channels,
            expected_number_of_channels);

    if (!data)
    {
        throw std::runtime_error("Failed to load image " + filepath);
    }

    if (number_of_channels != expected_number_of_channels)
    {
        std::cerr << "Encountered different number of channels than expected with texture " << filepath << "!\n";
    }
}

TextureData::~TextureData()
{
    stbi_image_free(data);
}
