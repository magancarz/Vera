#pragma once

#include "Assets/Defines.h"
#include "Vertex.h"

struct ModelInfo
{
    std::string name{Assets::DEFAULT_MODEL_NAME};
    std::string required_material{Assets::DEFAULT_MATERIAL_NAME};
    std::vector<Vertex> vertices{};
    std::vector<uint32_t> indices{};
};