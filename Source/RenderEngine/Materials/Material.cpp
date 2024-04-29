#include "Material.h"

Material::Material(Device& device, MaterialInfo in_material_info)
    : device{device}, material_info{in_material_info}
{
    createMaterialBuffer();
}

void Material::assignMaterialHitGroup(BlasInstance& blas_instance)
{
    //TODO: change it ofc
    uint32_t shader_off = 0;
    if (material_info.brightness > 0)
    {
//        shader_off = 2;
    }
    else if (material_info.fuzziness >= 0)
    {
//        shader_off = 2;
    }
    else if (material_info.refractive_index >= 0)
    {
//        shader_off = 3;
    }
    blas_instance.bottomLevelAccelerationStructureInstance.instanceShaderBindingTableRecordOffset = 0;//shader_off;
}

void Material::createMaterialBuffer()
{
    material_info_buffer = std::make_unique<Buffer>
    (
        device,
        sizeof(MaterialInfo),
        1,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    material_info_buffer->writeWithStagingBuffer(&material_info);
}

void Material::getMaterialDescription(ObjectDescription& object_description)
{
    object_description.material_address = material_info_buffer->getBufferDeviceAddress();
}