#include "World.h"

World::World(Window& window)
    : window{window}
{
    viewer_object = std::make_unique<Object>(Object::createObject());
    viewer_object->transform_component.translation.z = 2.5f;
    viewer_object->transform_component.rotation.y = glm::radians(180.f);

    camera.setPerspectiveProjection(glm::radians(50.0f), window.getAspect(), 0.1f, 100.f);
}

void World::loadObjects(Device& device/*, const std::vector<std::shared_ptr<Material>>& available_materials*/)
{
    std::shared_ptr<Model> monkey_model = Model::createModelFromFile(device, "Resources/Models/monkey.obj");
    auto left_monkey = std::make_shared<Object>(Object::createObject());
    left_monkey->model = monkey_model;
    left_monkey->color = {0.1f, 0.8f, 0.1f};
//    left_monkey->transform_component.translation = {-1.5f, .0f, 0};
//    left_monkey->transform_component.rotation = {.0f, .0f, glm::radians(180.0f)};
//    left_monkey->transform_component.scale = {.5f, .5f, .5f};
//    left_monkey->material = available_materials[0];
    objects.emplace(left_monkey->getID(), std::move(left_monkey));

//    auto right_monkey = std::make_shared<Object>(Object::createObject());
//    right_monkey->model = monkey_model;
//    right_monkey->color = {1.f, 1.f, 1.f};
//    right_monkey->transform_component.translation = {1.5f, .0f, 0};
//    right_monkey->transform_component.rotation = {.0f, 0.f, glm::radians(180.0f)};
//    right_monkey->transform_component.scale = {.5f, .5f, .5f};
////    right_monkey->material = available_materials[1];
//    objects.emplace(right_monkey->getID(), std::move(right_monkey));
//
//    std::shared_ptr<Model> plane_model = Model::createModelFromFile(device, "Resources/Models/plane.obj");
//    auto plane = std::make_shared<Object>(Object::createObject());
//    plane->model = plane_model;
//    plane->color = {0.1f, 0.8f, 0.1f};
//    plane->transform_component.translation = {0.f, 1.f, 0};
//    plane->transform_component.rotation = {.0f, 0.f, glm::radians(180.0f)};
//    plane->transform_component.scale = {3.f, 3.f, 3.f};
////    plane->material = available_materials[0];
//    objects.emplace(plane->getID(), std::move(plane));
//
//    std::vector<glm::vec3> light_colors
//    {
//        {1.f, .1f, .1f},
//        {.1f, .1f, 1.f},
//        {.1f, 1.f, .1f},
//        {1.f, 1.f, .1f},
//        {.1f, 1.f, 1.f},
//        {1.f, 1.f, 1.f}
//    };
//
//    for (int i = 0; i < light_colors.size(); ++i)
//    {
//        auto point_light = std::make_shared<Object>(Object::createPointLight(0.2f));
//        point_light->color = light_colors[i];
//        auto rotate_light = glm::rotate(
//                glm::mat4(1.f),
//                (i * glm::two_pi<float>()) / light_colors.size(),
//                {0.f, -1.f, 0.f});
//        point_light->transform_component.translation = glm::vec3(rotate_light * glm::vec4(-1.f, -1.f, -1.f, 1.f));
//        objects.emplace(point_light->getID(), std::move(point_light));
//    }
}

void World::update(FrameInfo& frame_info)
{
    movement_controller.moveInPlaneXZ(window.getGFLWwindow(), frame_info.frame_time, *viewer_object);
    camera.setViewYXZ(viewer_object->transform_component.translation, viewer_object->transform_component.rotation);
    frame_info.camera = &camera;
    frame_info.objects = objects;
}