#pragma once

#include <glm/glm.hpp>

#include "RenderEngine/Memory/Buffer.h"
#include "RenderEngine/Models/ObjectDescription.h"
#include "RenderEngine/Models/BlasInstance.h"
#include "MaterialInfo.h"
#include "MaterialDescription.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"

class Material
{
public:
    Material(MaterialInfo in_material_info, std::string material_name, std::shared_ptr<Texture> diffuse_texture, std::shared_ptr<Texture> normal_texture = nullptr);

    [[nodiscard]] std::string getName() const { return name; }
    [[nodiscard]] MaterialInfo getMaterialInfo() const { return material_info; }
    std::shared_ptr<Texture> getDiffuseTexture() { return diffuse_texture; }
    std::shared_ptr<Texture> getNormalTexture() { return normal_texture; }

    [[nodiscard]] bool isOpaque() const { return diffuse_texture->isOpaque(); }

protected:
    MaterialInfo material_info;
    std::string name;

    std::shared_ptr<Texture> diffuse_texture;
    std::shared_ptr<Texture> normal_texture;
    std::unique_ptr<Buffer> material_info_buffer;
};