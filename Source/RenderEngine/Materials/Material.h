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
    Material(MaterialInfo in_material_info, std::string material_name, std::shared_ptr<Texture> texture);

    [[nodiscard]] std::string getName() const { return name; }
    [[nodiscard]] MaterialInfo getMaterialInfo() const { return material_info; }
    std::shared_ptr<Texture> getTexture() { return texture; }

protected:
    MaterialInfo material_info;
    std::string name;

    std::shared_ptr<Texture> texture;
    std::unique_ptr<Buffer> material_info_buffer;
    uint32_t material_hit_group_index{0};

    friend class MaterialBuilder;
};