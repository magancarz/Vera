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

    camera.setPerspectiveProjection(glm::radians(50.0f), window.getAspect(), 0.1f, 100.f);
}

void World::loadObjects(Device& device, const std::vector<std::shared_ptr<Material>>& available_materials)
{
    std::shared_ptr<Model> plane_model = Model::createModelFromFile(device, "Resources/Models/plane.obj");
    std::shared_ptr<Model> cube_model = Model::createModelFromFile(device, "Resources/Models/cube.obj");
    std::shared_ptr<Model> dragon_model = Model::createModelFromFile(device, "Resources/Models/dragon.obj");
    std::map<std::string, std::shared_ptr<Model>> models;
    models.emplace("plane", plane_model);
    models.emplace("cube", cube_model);
    models.emplace("dragon", dragon_model);

    auto white_lambertian = std::make_shared<Material>(device, MaterialInfo{.color = glm::vec3{0.8f, 0.8f, 0.8f}});
    auto red_lambertian = std::make_shared<Material>(device, MaterialInfo{.color = glm::vec3{0.8f, 0.f, 0.f}});
    auto green_lambertian = std::make_shared<Material>(device, MaterialInfo{.color = glm::vec3{0.f, 0.8f, 0.f}});
    auto diffuse_light = std::make_shared<Material>(device, MaterialInfo{.color = glm::vec3{1.f, 1.f, 1.f}, .brightness = 3});
    auto specular = std::make_shared<Material>(device, MaterialInfo{.color = glm::vec3{1.f, 1.f, 1.f}, .refractive_index = 2.41});

    std::map<std::string, std::shared_ptr<Material>> materials;
    materials.emplace("white", white_lambertian);
    materials.emplace("red", red_lambertian);
    materials.emplace("light", diffuse_light);
    materials.emplace("specular", specular);

    ProjectInfo project_info = ProjectUtils::loadProject("vera");
    for (auto& object_info : project_info.objects_infos)
    {
        auto new_object = std::make_shared<Object>(Object::createObject());
        new_object->transform_component.translation = object_info.position;
        new_object->transform_component.rotation = object_info.rotation;
        new_object->transform_component.scale = glm::vec3{object_info.scale};
        new_object->setModel(models[object_info.model_name]);
        new_object->setMaterial(materials[object_info.material_name]);
        new_object->createBlasInstance();
        objects.emplace(new_object->getID(), std::move(new_object));
    }
}

void World::update(FrameInfo& frame_info)
{
    movement_controller.moveInPlaneXZ(window.getGFLWwindow(), frame_info.frame_time, *viewer_object);
    camera.setViewYXZ(viewer_object->transform_component.translation, viewer_object->transform_component.rotation);
    frame_info.player_moved = movement_controller.playerMoved();
    frame_info.camera = &camera;
}