#pragma once

#include <string>

class DeviceTexture;

struct MaterialInfo
{
    std::string name;
    DeviceTexture* diffuse_texture{nullptr};
    DeviceTexture* normal_texture{nullptr};
};