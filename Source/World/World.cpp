#include <iostream>
#include "World.h"
#include "Project/Project.h"
#include "Objects/Components/CameraComponent.h"

World::World(Window& window)
    : window{window} {}

void World::loadObjects(const ProjectInfo& project_info, const std::shared_ptr<AssetManager>& asset_manager)
{
    loadViewerObject();

    for (auto& object_info : project_info.objects_infos)
    {
        auto new_object = std::make_shared<Object>(Object::createObject());
        auto transform_component = std::make_shared<TransformComponent>(new_object.get(), this);
        transform_component->translation = object_info.position;
        transform_component->rotation = object_info.rotation;
        transform_component->scale = glm::vec3{object_info.scale};
        new_object->addComponent(std::move(transform_component));
        new_object->setModel(asset_manager->fetchModel(object_info.model_name));
        new_object->setMaterial(asset_manager->fetchMaterial(object_info.material_name));
        new_object->createBlasInstance();
        rendered_objects.emplace(new_object->getID(), std::move(new_object));
    }
}

void World::loadViewerObject()
{
    viewer_object = std::make_shared<Object>(Object::createObject());
    auto transform_component = std::make_shared<TransformComponent>(viewer_object.get(), this);
    transform_component->translation.y = 5.f;
    transform_component->translation.z = 10.f;
    transform_component->rotation.y = glm::radians(180.f);
    auto player_movement_component = std::make_shared<PlayerMovementComponent>(viewer_object.get(), this, transform_component);
    viewer_object->addComponent(std::move(player_movement_component));
    auto player_camera_component = std::make_shared<CameraComponent>(viewer_object.get(), this, transform_component);
    player_camera_component->setPerspectiveProjection(glm::radians(70.0f), window.getAspect(), 0.1f, 100.f);
    viewer_object->addComponent(std::move(player_camera_component));
    viewer_object->addComponent(std::move(transform_component));
}

void World::update(FrameInfo& frame_info)
{
    removeUnusedRegisteredComponents();
    updateRegisteredComponents(frame_info);
}

void World::removeUnusedRegisteredComponents()
{
    for (auto& [_, tick_group_components] : registered_components)
    {
        tick_group_components.erase(std::remove_if(
                tick_group_components.begin(), tick_group_components.end(),
                [](ObjectComponent* ptr) { return ptr == nullptr; }), tick_group_components.end());
    }
}

void World::updateRegisteredComponents(FrameInfo& frame_info)
{
    for (auto& [_, tick_group_components] : registered_components)
    {
        for (auto& component : tick_group_components)
        {
            assert(component && "Registered component cannot be nullptr! All components with non existent owners should be cleaned before");
            component->update(frame_info);
        }
    }
}

void World::registerComponent(ObjectComponent* component)
{
    assert(component && "Cannot register component which is nullptr!");
    registered_components[component->getTickGroup()].emplace_back(component);
}