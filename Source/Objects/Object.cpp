#include "Object.h"

#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "Objects/Components/TransformComponent.h"
#include "Objects/Components/MeshComponent.h"

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

glm::mat4 Object::getTransform()
{
    if (!transform_component_cache)
    {
        transform_component_cache = findComponentByClass<TransformComponent>();
    }

    if (transform_component_cache)
    {
        return transform_component_cache->transform();
    }

    return glm::mat4{1};
}

void Object::addComponent(std::shared_ptr<ObjectComponent> component)
{
    assert(component->getOwner() == this && "Component's owner should be the same as the object to which component tried to be added");
    components.emplace_back(std::move(component));
}
