#include "Scene.h"

#include "editor/Project/Project.h"
#include "Models/AssetManager.h"
#include "Objects/Object.h"
#include "Objects/ShapesCollector.h"
#include "Materials/MaterialAsset.h"
#include "Materials/Material.h"
#include "RenderEngine/RayTracing/IntersectionAccelerators/BVHTreeBuilder.h"
#include "Objects/Lights/Light.h"
#include "Objects/Lights/PointLight.h"
#include "Objects/Lights/DirectionalLight.h"
#include "Objects/Lights/Spotlight.h"
#include "Objects/TriangleMesh.h"

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
    objects.clear();
    lights.clear();

    for (const auto& object_info_as_string : project_info.objects_infos)
    {
        ObjectInfo object_info = ObjectInfo::fromString(object_info_as_string);
        auto material = AssetManager::findMaterialAsset(object_info.material_name);
        auto model = AssetManager::findModelAsset(object_info.model_name);

        if (material->cuda_material->isEmittingLight())
        {
            std::shared_ptr<Light> light = std::make_shared<PointLight>(
                    this, object_info.position, glm::vec3{1, 1, 1}, glm::vec3{1, 0.01, 0.0001});
            objects.push_back(light);
            lights.push_back(light);
        }
        else
        {
            auto object = std::make_shared<TriangleMesh>(this, material, model, object_info.position, object_info.rotation, object_info.scale);
            object->createShapesForRayTracedMesh();
		    objects.push_back(object);
		    triangle_meshes.push_back(object);
        }
    }

    need_to_build_intersection_accelerator = true;
}

void Scene::createTriangleMesh(std::shared_ptr<RawModel> model)
{
    const auto triangle_mesh = std::make_shared<TriangleMesh>(this, AssetManager::findMaterialAsset("white"), std::move(model), glm::vec3{0}, glm::vec3{0}, 1.f);
    triangle_mesh->createShapesForRayTracedMesh();
    objects.push_back(triangle_mesh);
    triangle_meshes.push_back(triangle_mesh);
    need_to_build_intersection_accelerator = true;
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

std::vector<std::string> Scene::gatherObjectsInfos()
{
    std::vector<std::string> objects_infos;
    objects_infos.reserve(objects.size());
    for (const auto& object : objects)
    {
        objects_infos.push_back(object->getObjectInfo());
    }
    return objects_infos;
}