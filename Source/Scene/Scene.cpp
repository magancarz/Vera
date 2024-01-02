#include "Scene.h"

#include "editor/Project/Project.h"
#include "Models/AssetManager.h"
#include "Objects/Object.h"
#include "Objects/ShapesCollector.h"
#include "Materials/MaterialAsset.h"
#include "Materials/Material.h"
#include "RenderEngine/RayTracing/IntersectionAccelerators/BVHTreeBuilder.h"
#include "Objects/Lights/Light.h"
#include "Objects/TriangleMesh.h"
#include "Shaders/StaticShader.h"

void Scene::notifyOnObjectChange()
{
    refreshScene();
    buildSceneIntersectionAccelerator();
}

void Scene::buildSceneIntersectionAcceleratorIfNeeded()
{
    if (need_to_build_intersection_accelerator)
    {
        buildSceneIntersectionAccelerator();
        need_to_build_intersection_accelerator = false;
    }
}

void Scene::notifyOnObjectMaterialChange()
{
    need_to_build_intersection_accelerator = true;
}

void Scene::deleteObject(const Object* scene_object)
{
    for (int i = 0; i < objects.size(); ++i)
    {
        if (objects[i]->object_id == scene_object->object_id)
        {
            objects.erase(objects.begin() + i);
            objects.shrink_to_fit();
        }
    }
    Algorithms::removeExpiredWeakPointers(triangle_meshes);
    Algorithms::removeExpiredWeakPointers(lights);
    need_to_build_intersection_accelerator = true;
}

std::weak_ptr<Object> Scene::findObjectByID(unsigned id)
{
    for (const auto& object : objects)
    {
        if (object->object_id == id)
        {
            return object;
        }
    }
    return {};
}

void Scene::refreshScene()
{
    need_to_build_intersection_accelerator = true;
}

void Scene::loadSceneFromProject(const ProjectInfo& project_info)
{
    loadTriangleMeshes(project_info.objects_infos);
    loadLights(project_info.lights_infos);

    need_to_build_intersection_accelerator = true;
}

void Scene::loadTriangleMeshes(const std::vector<std::string>& triangle_meshes_infos)
{
    objects.clear();
    for (const auto& triangle_mesh_info_as_string : triangle_meshes_infos)
    {
        TriangleMeshInfo triangle_mesh_info = TriangleMeshInfo::fromString(triangle_mesh_info_as_string);
        auto material = AssetManager::findMaterialAsset(triangle_mesh_info.material_name);
        auto model = AssetManager::findModelAsset(triangle_mesh_info.model_name);
        auto triangle_mesh = std::make_shared<TriangleMesh>(this, material, model, triangle_mesh_info.position, triangle_mesh_info.rotation, triangle_mesh_info.scale);
        triangle_mesh->createShapesForRayTracedMesh();
        objects.push_back(triangle_mesh);
        triangle_meshes.push_back(triangle_mesh);
    }
}

void Scene::loadLights(const std::vector<std::string>& lights_infos)
{
    lights.clear();
    for (const auto& light_info_as_string : lights_infos)
    {
        for (const auto& light_creator : AssetManager::getAvailableLightCreators())
        {
            if (light_creator->apply(light_info_as_string))
            {
                std::shared_ptr<Light> light = light_creator->fromLightInfo(this, light_info_as_string);
                objects.push_back(light);
                lights.push_back(light);
                break;
            }
        }
    }
}

void Scene::createTriangleMesh(std::shared_ptr<RawModel> model)
{
    const auto triangle_mesh = std::make_shared<TriangleMesh>(this, AssetManager::findMaterialAsset("white"), std::move(model), glm::vec3{0}, glm::vec3{0}, 1.f);
    triangle_mesh->createShapesForRayTracedMesh();
    objects.push_back(triangle_mesh);
    triangle_meshes.push_back(triangle_mesh);
    need_to_build_intersection_accelerator = true;
}

void Scene::createSceneLight(const std::shared_ptr<LightCreator>& light_creator)
{
    if (lights.size() < StaticShader::MAX_LIGHTS)
    {
        std::shared_ptr<Light> light = light_creator->create(this);
        objects.push_back(light);
        lights.push_back(light);
    }
    else
    {
        std::cerr << "Couldn't create new light, because number of lights is already at max limit!\n";
    }
}

void Scene::buildSceneIntersectionAccelerator()
{
    ShapesCollector shapes_collector{triangle_meshes};
    const CollectedShapes collected_shapes = shapes_collector.collectShapes();
    scene_light_sources = dmm::DeviceMemoryPointer<Shape*>{collected_shapes.number_of_light_emitting_shapes};
    scene_light_sources.copyFrom(collected_shapes.light_emitting_shapes.data());
    bvh_tree_builder = std::make_unique<BVHTreeBuilder>(collected_shapes);
    intersection_accelerator_tree_traverser = bvh_tree_builder->buildAccelerator();
}

std::vector<std::string> Scene::gatherTriangleMeshesInfos()
{
    std::vector<std::string> objects_infos;
    objects_infos.reserve(triangle_meshes.size());
    for (const auto& object : triangle_meshes)
    {
        objects_infos.push_back(object.lock()->getObjectInfo());
    }
    return objects_infos;
}

std::vector<std::string> Scene::gatherLightsInfos()
{
    std::vector<std::string> lights_infos;
    lights_infos.reserve(lights.size());
    for (const auto& light : lights)
    {
        lights_infos.push_back(light.lock()->getObjectInfo());
    }
    return lights_infos;
}