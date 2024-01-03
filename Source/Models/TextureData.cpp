#include "TextureData.h"

#include "stb_image.h"

TextureData::~TextureData()
{
    stbi_image_free(texture_data);
}