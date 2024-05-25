#pragma once

#include "ModelDescription.h"

struct MeshDescription
{
    std::vector<ModelDescription> model_descriptions;
    std::vector<std::string> required_materials;
};