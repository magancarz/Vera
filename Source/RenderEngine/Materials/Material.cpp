#include "Material.h"

Material::Material(MaterialInfo in_material_info, std::string material_name)
    : material_info{in_material_info}, name{std::move(material_name)} {}

MaterialDescription Material::getMaterialDescription()
{
    return
    {
            .material_hit_group_index = material_hit_group_index,
            .material_info_buffer_device_address = material_info_buffer->getBufferDeviceAddress(),
            .material_texture_layout = texture->getImageLayout(),
            .material_texture_view = texture->getImageView(),
            .material_texture_sampler = texture->getSampler()
    };
}