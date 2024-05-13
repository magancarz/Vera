#include <iostream>
#include "World.h"
#include "Project/Project.h"

World::World(Window& window)
    : window{window}
{
    viewer_object = std::make_unique<Object>(Object::createObject());
    viewer_object->transform_component.translation.y = 5.f;
    viewer_object->transform_component.translation.z = 10.f;
    viewer_object->transform_component.rotation.y = glm::radians(180.f);

    camera.setPerspectiveProjection(glm::radians(70.0f), window.getAspect(), 0.1f, 100.f);
}

void World::loadObjects(const ProjectInfo& project_info, AssetManager& asset_manager)
{
    for (auto& object_info : project_info.objects_infos)
    {
        auto new_object = std::make_shared<Object>(Object::createObject());
        new_object->transform_component.translation = object_info.position;
        new_object->transform_component.rotation = object_info.rotation;
        new_object->transform_component.scale = glm::vec3{object_info.scale};
        new_object->setModel(asset_manager.fetchModel(object_info.model_name));
        new_object->setMaterial(asset_manager.fetchMaterial(object_info.material_name));
        new_object->createBlasInstance();
        objects.emplace(new_object->getID(), std::move(new_object));
    }
}

void World::update(FrameInfo& frame_info)
{
    movement_controller.moveInPlaneXZ(window.getGFLWwindow(), frame_info.frame_time, *viewer_object);
    camera.setViewYXZ(viewer_object->transform_component.translation, viewer_object->transform_component.rotation);
    frame_info.need_to_refresh_generated_image = movement_controller.playerMoved();
    frame_info.camera = &camera;
}