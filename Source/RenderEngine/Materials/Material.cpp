#include "Material.h"

Material::Material(MaterialInfo in_material_info, std::string material_name, std::shared_ptr<Texture> texture)
    : material_info{in_material_info}, name{std::move(material_name)}, texture{std::move(texture)} {}
