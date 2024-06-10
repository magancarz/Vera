#pragma once

#include "TextureData.h"

class DeviceTexture;

class TextureLoader
{
public:
    static TextureData loadFromAssetFile(const std::string& texture_name);

    static constexpr int EXPECTED_NUMBER_OF_CHANNELS = 4;
};
