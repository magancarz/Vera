#pragma once

#include "Models/ModelData.h"

class OBJLoader {
public:
    static ModelData loadModelDataFromOBJFileFormat(const std::string& file_name);
};