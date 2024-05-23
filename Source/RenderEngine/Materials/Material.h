#pragma once

#include <glm/glm.hpp>

#include "RenderEngine/RenderingAPI/Buffer.h"
#include "RenderEngine/Models/ObjectDescription.h"
#include "RenderEngine/Models/BlasInstance.h"
#include "MaterialInfo.h"
#include "MaterialDescription.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"

class Material
{
public:
    Material(MaterialInfo in_material_info, std::string material_name);

    [[nodiscard]] std::string getName() const { return name; }

    MaterialDescription getMaterialDescription();
    [[nodiscard]] bool isLightMaterial() const { return material_info.brightness > 0; }

    //TODO: temp
    std::shared_ptr<Texture> getTexture() { return texture; }

protected:
    MaterialInfo material_info;
    std::string name;

    std::shared_ptr<Texture> texture;
    std::unique_ptr<Buffer> material_info_buffer;
    uint32_t material_hit_group_index{0};

    friend class MaterialBuilder;
};