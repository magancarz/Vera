#pragma once

#include "MaterialData.h"
#include "Assets/AssetManager.h"

class VeraMaterialLoader
{
public:
    static MaterialData loadAssetFromFile(const std::string& asset_name);

    inline static const char* VERA_MATERIAL_FILE_EXTENSION{".mat"};
};
