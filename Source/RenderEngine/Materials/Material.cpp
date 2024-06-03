#include "Material.h"

Material::Material(const MaterialInfo& in_material_info, std::string material_name, Texture* diffuse_texture, Texture* normal_texture)
    : material_info{in_material_info}, name{std::move(material_name)}, diffuse_texture{diffuse_texture}, normal_texture{normal_texture} {}
