#include "World.h"

#include "Editor/Window/WindowSystem.h"
#include "Project/Project.h"
#include "Objects/Components/CameraComponent.h"
#include "Objects/Components/MeshComponent.h"
#include "Objects/Components/TransformComponent.h"
#include "Objects/Object.h"

void World::loadProject(const ProjectInfo& project_info, AssetManager& asset_manager)
{
    for (auto& object_info : project_info.objects_infos)
    {
        auto new_object = std::make_unique<Object>();
        auto transform_component = std::make_unique<TransformComponent>(*new_object);
        transform_component->translation = object_info.position;
        transform_component->rotation = object_info.rotation;
        transform_component->scale = glm::vec3{object_info.scale};
        new_object->addRootComponent(std::move(transform_component));
        auto mesh_component = std::make_unique<MeshComponent>(*new_object);
        mesh_component->setMesh(asset_manager.fetchMesh(object_info.mesh_name));
        new_object->addComponent(std::move(mesh_component));
        storeObject(std::move(new_object));
    }
}

Object* World::storeObject(std::unique_ptr<Object> object)
{
    uint32_t object_id = object->getID();
    objects.emplace(object_id, std::move(object));
    return objects[object_id].get();
}

void World::createViewerObject(InputManager& input_manager)
{
    auto new_viewer_object = std::make_unique<Object>();
    auto transform_component = std::make_unique<TransformComponent>(*new_viewer_object);
    transform_component->translation.y = 5.f;
    transform_component->translation.z = 10.f;
    transform_component->rotation.y = glm::radians(180.f);
    auto player_movement_component = std::make_unique<PlayerMovementComponent>(*new_viewer_object, input_manager, transform_component.get());
    new_viewer_object->addComponent(std::move(player_movement_component));
    auto player_camera_component = std::make_unique<CameraComponent>(*new_viewer_object, transform_component.get());
    player_camera_component->setPerspectiveProjection(glm::radians(70.0f), WindowSystem::get().getAspect(), 0.1f, 100.f);
    new_viewer_object->addComponent(std::move(player_camera_component));
    new_viewer_object->addRootComponent(std::move(transform_component));
    viewer_object = storeObject(std::move(new_viewer_object));
}

void World::update(FrameInfo& frame_info)
{
    for (auto& [id, object] : objects)
    {
        object->update(frame_info);
    }
}
