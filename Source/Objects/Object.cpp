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

void Object::createBlasInstance()
{
    assert(model != nullptr);

    blas_instance = model->createBlasInstance(transform_component.transform(), id);
}