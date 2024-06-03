#include "Object.h"

#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "Objects/Components/TransformComponent.h"
#include "Objects/Components/MeshComponent.h"

Object::Object()
    : id{available_id++} {}

glm::vec3 Object::getLocation()
{
    if (root_component)
    {
        return root_component->translation;
    }

    return glm::vec3{0};
}

glm::mat4 Object::getTransform()
{
    if (root_component)
    {
        return root_component->transform();
    }

    return glm::mat4{1};
}

void Object::addComponent(std::shared_ptr<ObjectComponent> component)
{
    assert(component->getOwner().getID() == this->getID() && "Component's owner should be the same as the object to which component tried to be added");
    components.emplace_back(std::move(component));
}

void Object::addRootComponent(std::shared_ptr<TransformComponent> transform_component)
{
    assert(transform_component && "It is useless to pass empty root component");
    addComponent(std::move(transform_component));
    root_component = findComponentByClass<TransformComponent>();
}
