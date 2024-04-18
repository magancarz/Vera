#include "World.h"

World::World(Window& window)
    : window{window}
{
    viewer_object = std::make_unique<Object>(Object::createObject());
    viewer_object->transform_component.translation.y = 5.f;
    viewer_object->transform_component.translation.z = 10.f;
    viewer_object->transform_component.rotation.y = glm::radians(180.f);

    camera.setPerspectiveProjection(glm::radians(50.0f), window.getAspect(), 0.1f, 100.f);
}

void World::loadObjects(Device& device, const std::vector<std::shared_ptr<Material>>& available_materials)
{
    std::shared_ptr<Model> cube_scene_model = Model::createModelFromFile(device, "Resources/Models/cube_scene.obj");
    auto cube_scene = std::make_shared<Object>(Object::createObject());
    cube_scene->setModel(cube_scene_model);
//    left_monkey->material = available_materials[0];
    objects.emplace(cube_scene->getID(), std::move(cube_scene));

    auto second_cube_scene = std::make_shared<Object>(Object::createObject());
    second_cube_scene->transform_component.translation.x = 10.f;
    second_cube_scene->setModel(cube_scene_model);
//    left_monkey->material = available_materials[0];
    objects.emplace(second_cube_scene->getID(), std::move(second_cube_scene));
}

void World::update(FrameInfo& frame_info)
{
    movement_controller.moveInPlaneXZ(window.getGFLWwindow(), frame_info.frame_time, *viewer_object);
    camera.setViewYXZ(viewer_object->transform_component.translation, viewer_object->transform_component.rotation);
    frame_info.player_moved = movement_controller.playerMoved();
    frame_info.camera = &camera;
    frame_info.objects = objects;
}