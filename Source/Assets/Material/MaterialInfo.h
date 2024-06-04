#pragma once

#include <string>

class Texture;

struct MaterialInfo
{
    std::string name;
    Texture* diffuse_texture;
    Texture* normal_texture;
};