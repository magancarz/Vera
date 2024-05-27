#include "Material.h"

Material::Material(MaterialInfo in_material_info, std::string material_name, std::shared_ptr<Texture> diffuse_texture, std::shared_ptr<Texture> normal_texture)
    : material_info{in_material_info}, name{std::move(material_name)}, diffuse_texture{std::move(diffuse_texture)}, normal_texture{std::move(normal_texture)} {}
