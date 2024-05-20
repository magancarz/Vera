#include "Object.h"

#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "Objects/Components/TransformComponent.h"

Object::Object()
    : id{available_id++} {}

glm::vec3 Object::getLocation()
{
    if (!transform_component_cache)
    {
        transform_component_cache = findComponentByClass<TransformComponent>();
    }

    if (transform_component_cache)
    {
        return transform_component_cache->translation;
    }

    return glm::vec3{0};
}

void Object::addComponent(std::shared_ptr<ObjectComponent> component)
{
    components.emplace_back(std::move(component));
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

    auto transform_component = findComponentByClass<TransformComponent>();
    blas_instance = model->createBlasInstance(transform_component->transform(), id);
    material->assignMaterialHitGroup(blas_instance);
}

ObjectDescription Object::getObjectDescription()
{
    auto transform_component = findComponentByClass<TransformComponent>();
    ObjectDescription object_description{};
    model->getModelDescription(object_description);
    material->getMaterialDescription(object_description);
    object_description.surface_area *= transform_component->scale.x * transform_component->scale.x;
    object_description.object_to_world = transform_component->transform();
    return object_description;
}