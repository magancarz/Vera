#include "Object.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

Object Object::createObject()
{
    static id_t available_id = 0;
    return Object{available_id++};
}

void Object::setModel(std::shared_ptr<Model> in_model)
{
    model = std::move(in_model);
}

void Object::setMaterial(std::shared_ptr<Material> in_material)
{
    material = std::move(in_material);
}

void Object::createBlasInstance()
{
    assert(model != nullptr && material != nullptr);

    blas_instance = model->createBlasInstance(transform_component.transform(), id);
    material->assignMaterialHitGroup(blas_instance);
}

ObjectDescription Object::getObjectDescription()
{
    ObjectDescription object_description{};
    model->getModelDescription(object_description);
    material->getMaterialDescription(object_description);
    object_description.surface_area *= transform_component.scale.x * transform_component.scale.x;
    object_description.object_to_world = transform_component.transform();
    return object_description;
}