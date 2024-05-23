#include <iostream>
#include "World.h"
#include "Project/Project.h"
#include "Objects/Components/CameraComponent.h"
#include "Objects/Components/MeshComponent.h"

void World::loadProject(const ProjectInfo& project_info, const std::shared_ptr<AssetManager>& asset_manager)
{
    asset_manager->loadNeededAssetsForProject(project_info);
    for (auto& object_info : project_info.objects_infos)
    {
        auto new_object = std::make_shared<Object>();
        auto transform_component = std::make_shared<TransformComponent>(new_object.get());
        registerComponent(transform_component);
        transform_component->translation = object_info.position;
        transform_component->rotation = object_info.rotation;
        transform_component->scale = glm::vec3{object_info.scale};
        new_object->addComponent(std::move(transform_component));
        auto mesh_component = std::make_shared<MeshComponent>(new_object.get());
        registerComponent(mesh_component);
        mesh_component->setModel(asset_manager->fetchModel(object_info.model_name));
        mesh_component->setMaterial(asset_manager->fetchMaterial(object_info.material_name));
        new_object->addComponent(std::move(mesh_component));
        rendered_objects.emplace(new_object->getID(), std::move(new_object));
    }
}

void World::createViewerObject(const std::shared_ptr<InputManager>& input_manager)
{
    viewer_object = std::make_shared<Object>();
    auto transform_component = std::make_shared<TransformComponent>(viewer_object.get());
    registerComponent(transform_component);
    transform_component->translation.y = 5.f;
    transform_component->translation.z = 10.f;
    transform_component->rotation.y = glm::radians(180.f);
    auto player_movement_component = std::make_shared<PlayerMovementComponent>(viewer_object.get(), input_manager, transform_component);
    registerComponent(player_movement_component);
    viewer_object->addComponent(std::move(player_movement_component));
    auto player_camera_component = std::make_shared<CameraComponent>(viewer_object.get(), transform_component);
    registerComponent(player_camera_component);
    player_camera_component->setPerspectiveProjection(glm::radians(70.0f), Window::get()->getAspect(), 0.1f, 100.f);
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
                [](const std::weak_ptr<ObjectComponent>& component) { return component.expired(); }), tick_group_components.end());
    }
}

void World::updateRegisteredComponents(FrameInfo& frame_info)
{
    for (auto& [_, tick_group_components] : registered_components)
    {
        for (auto& component : tick_group_components)
        {
            assert(!component.expired() && "Registered component cannot be nullptr! All components with non existent owners should be cleaned before");
            component.lock()->update(frame_info);
        }
    }
}

void World::registerComponent(std::weak_ptr<ObjectComponent> component)
{
    assert(!component.expired() && "Cannot register component which is nullptr!");
    registered_components[component.lock()->getTickGroup()].emplace_back(std::move(component));
}