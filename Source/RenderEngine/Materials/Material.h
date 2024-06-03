#pragma once

#include "RenderEngine/Memory/Buffer.h"
#include "MaterialInfo.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"

class Material
{
public:
    Material(const MaterialInfo& in_material_info, std::string material_name, Texture* diffuse_texture, Texture* normal_texture = nullptr);

    [[nodiscard]] std::string getName() const { return name; }
    [[nodiscard]] MaterialInfo getMaterialInfo() const { return material_info; }
    [[nodiscard]] Texture* getDiffuseTexture() const { return diffuse_texture; }
    [[nodiscard]] Texture* getNormalTexture() const { return normal_texture; }

    [[nodiscard]] bool isOpaque() const { return diffuse_texture->isOpaque(); }

protected:
    MaterialInfo material_info{};
    std::string name{};

    Texture* diffuse_texture{nullptr};
    Texture* normal_texture{nullptr};
    std::unique_ptr<Buffer> material_info_buffer{nullptr};
};