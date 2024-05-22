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

ObjectDescription Object::getObjectDescription()
{
    auto mesh_component = findComponentByClass<MeshComponent>();
    auto transform_component = findComponentByClass<TransformComponent>();
    assert(mesh_component && transform_component && "To obtain object description mesh component and transform component must be present in object components list!");

    ObjectDescription object_description{};
    ModelDescription model_description = mesh_component->getModel()->getModelDescription();
    object_description.vertex_address = model_description.vertex_address;
    object_description.index_address = model_description.index_address;
    object_description.num_of_triangles = model_description.num_of_triangles;

    MaterialDescription material_description = mesh_component->getMaterial()->getMaterialDescription();
    object_description.material_address = material_description.material_info_buffer_device_address;

    object_description.surface_area *= transform_component->scale.x * transform_component->scale.x;
    object_description.object_to_world = transform_component->transform();
    return object_description;
}