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
    std::shared_ptr<Model> plane_model = Model::createModelFromFile(device, "Resources/Models/plane.obj");
    std::shared_ptr<Model> cube_model = Model::createModelFromFile(device, "Resources/Models/dragon.obj");

    auto white_lambertian = std::make_shared<Material>(device, MaterialInfo{.color = glm::vec3{0.8f, 0.8f, 0.8f}});
    auto red_lambertian = std::make_shared<Material>(device, MaterialInfo{.color = glm::vec3{0.8f, 0.f, 0.f}});
    auto green_lambertian = std::make_shared<Material>(device, MaterialInfo{.color = glm::vec3{0.f, 0.8f, 0.f}});
    auto diffuse_light = std::make_shared<Material>(device, MaterialInfo{.color = glm::vec3{1.f, 1.f, 1.f}, .brightness = 1});

    auto bottom_plane = std::make_shared<Object>(Object::createObject());
    bottom_plane->transform_component.scale = glm::vec3{3.f};
    bottom_plane->setModel(plane_model);
    bottom_plane->setMaterial(white_lambertian);
    objects.emplace(bottom_plane->getID(), std::move(bottom_plane));

    auto top_plane = std::make_shared<Object>(Object::createObject());
    top_plane->transform_component.translation.y = 6.f;
    top_plane->transform_component.rotation.x = glm::radians(180.f);
    top_plane->transform_component.scale = glm::vec3{3.f};
    top_plane->setModel(plane_model);
    top_plane->setMaterial(white_lambertian);
    objects.emplace(top_plane->getID(), std::move(top_plane));

//    auto front_plane = std::make_shared<Object>(Object::createObject());
//    front_plane->transform_component.translation.y = 3.f;
//    front_plane->transform_component.translation.z = 3.f;
//    front_plane->transform_component.rotation.x = glm::radians(90.f);
//    front_plane->transform_component.rotation.y = glm::radians(180.f);
//    front_plane->transform_component.scale = glm::vec3{3.f};
//    front_plane->setModel(plane_model);
//    front_plane->setMaterial(white_lambertian);
//    objects.emplace(front_plane->getID(), std::move(front_plane));

    auto back_plane = std::make_shared<Object>(Object::createObject());
    back_plane->transform_component.translation.y = 3.f;
    back_plane->transform_component.translation.z = -3.f;
    back_plane->transform_component.rotation.x = glm::radians(90.f);
    back_plane->transform_component.scale = glm::vec3{3.f};
    back_plane->setModel(plane_model);
    back_plane->setMaterial(white_lambertian);
    objects.emplace(back_plane->getID(), std::move(back_plane));

    auto left_plane = std::make_shared<Object>(Object::createObject());
    left_plane->transform_component.translation.x = 3.f;
    left_plane->transform_component.translation.y = 3.f;
    left_plane->transform_component.rotation.x = glm::radians(90.f);
    left_plane->transform_component.rotation.y = glm::radians(-90.f);
    left_plane->transform_component.scale = glm::vec3{3.f};
    left_plane->setModel(plane_model);
    left_plane->setMaterial(red_lambertian);
    objects.emplace(left_plane->getID(), std::move(left_plane));

    auto right_plane = std::make_shared<Object>(Object::createObject());
    right_plane->transform_component.translation.x = -3.f;
    right_plane->transform_component.translation.y = 3.f;
    right_plane->transform_component.rotation.x = glm::radians(90.f);
    right_plane->transform_component.rotation.y = glm::radians(90.f);
    right_plane->transform_component.scale = glm::vec3{3.f};
    right_plane->setModel(plane_model);
    right_plane->setMaterial(green_lambertian);
    objects.emplace(right_plane->getID(), std::move(right_plane));

    auto light_plane = std::make_shared<Object>(Object::createObject());
    light_plane->transform_component.translation.y = 5.999f;
    light_plane->transform_component.scale = glm::vec3{.5f};
    light_plane->setModel(plane_model);
    light_plane->setMaterial(diffuse_light);
    objects.emplace(light_plane->getID(), std::move(light_plane));

    auto dragon = std::make_shared<Object>(Object::createObject());
    dragon->transform_component.translation.y = 2.f;
    dragon->transform_component.scale = glm::vec3{0.25f};
    dragon->setModel(cube_model);
    dragon->setMaterial(white_lambertian);
    objects.emplace(dragon->getID(), std::move(dragon));
}

void World::update(FrameInfo& frame_info)
{
    movement_controller.moveInPlaneXZ(window.getGFLWwindow(), frame_info.frame_time, *viewer_object);
    camera.setViewYXZ(viewer_object->transform_component.translation, viewer_object->transform_component.rotation);
    frame_info.player_moved = movement_controller.playerMoved();
    frame_info.camera = &camera;
    frame_info.objects = objects;
}