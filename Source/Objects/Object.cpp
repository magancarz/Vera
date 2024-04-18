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
    createBlasInstance();
}

void Object::setMaterial(std::shared_ptr<Material> in_material)
{
    material = std::move(in_material);
}

void Object::createBlasInstance()
{
    assert(model != nullptr);

    blas_instance = model->createBlasInstance(transform_component.transform(), id);
}

ObjectDescription Object::getObjectDescription() const
{
    ObjectDescription object_description{};
    model->getModelDescription(object_description);
    material->getMaterialDescription(object_description);
    return object_description;
}