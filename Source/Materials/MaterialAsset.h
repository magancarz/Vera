#pragma once

#include <memory>

#include "Utils/DeviceMemoryPointer.h"
#include "TextureAsset.h"

struct MaterialAsset
{
    MaterialAsset(
		dmm::DeviceMemoryPointer<Material> cuda_material,
		std::shared_ptr<Material> material_ptr,
		std::shared_ptr<TextureAsset> texture,
		std::shared_ptr<TextureAsset> normal_map_texture,
		std::shared_ptr<TextureAsset> specular_map_texture,
		std::shared_ptr<TextureAsset> depth_map_texture)
    : cuda_material(std::move(cuda_material)),
    material(std::move(material_ptr)),
    texture(std::move(texture)),
    normal_map_texture(std::move(normal_map_texture)),
    specular_map_texture(std::move(specular_map_texture)),
    depth_map_texture(std::move(depth_map_texture)) {}

    dmm::DeviceMemoryPointer<Material> cuda_material;
    std::shared_ptr<Material> material;
    std::shared_ptr<TextureAsset> texture;
    std::shared_ptr<TextureAsset> normal_map_texture;
    std::shared_ptr<TextureAsset> specular_map_texture;
    std::shared_ptr<TextureAsset> depth_map_texture;
};
