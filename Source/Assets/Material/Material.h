#pragma once

#include "MaterialInfo.h"
#include "RenderEngine/Textures/DeviceTexture.h"

class MemoryAllocator;

class Material
{
public:
    Material(const MaterialInfo& in_material_info);

    [[nodiscard]] std::string getName() const { return name; }
    [[nodiscard]] DeviceTexture* getDiffuseTexture() const { return diffuse_texture; }
    [[nodiscard]] DeviceTexture* getNormalTexture() const { return normal_texture; }

    [[nodiscard]] bool isOpaque() const { return diffuse_texture->isOpaque(); }

protected:
    std::string name{};

    DeviceTexture* diffuse_texture{nullptr};
    DeviceTexture* normal_texture{nullptr};
};
