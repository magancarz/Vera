#pragma once

#include "Assets/AssetManager.h"

class VeraMaterialLoader
{
public:
    static void loadAssetFromFile(AssetManager& asset_manager, const std::string& asset_name);

    inline static const char* VERA_MATERIAL_FILE_EXTENSION{".mat"};
};
