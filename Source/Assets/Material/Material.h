#pragma once

#include "MaterialInfo.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"

class MemoryAllocator;

class Material
{
public:
    Material(const MaterialInfo& in_material_info);

    [[nodiscard]] std::string getName() const { return name; }
    [[nodiscard]] Texture* getDiffuseTexture() const { return diffuse_texture; }
    [[nodiscard]] Texture* getNormalTexture() const { return normal_texture; }

    [[nodiscard]] bool isOpaque() const { return diffuse_texture->isOpaque(); }

protected:
    std::string name{};

    Texture* diffuse_texture{nullptr};
    Texture* normal_texture{nullptr};
};
