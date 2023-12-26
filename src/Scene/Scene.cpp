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

void Scene::notifyOnObjectMaterialChange(const Object* object)
{
    if (object->isEmittingSomeLight() && !isObjectAlreadySampled(object->object_id))
    {
        const std::weak_ptr<Object> object_as_weak_ptr = findObjectByID(object->object_id);
        lights.push_back(object_as_weak_ptr);
    }
    need_to_build_intersection_accelerator = true;
}

void Scene::deleteObject(const Object* scene_object)
{
    for (int i = 0; i < objects.size(); ++i)
    {
        if (objects[i]->object_id == scene_object->object_id)
        {
            if (scene_object->isEmittingSomeLight())
            {
                for (int i = 0; i < lights.size(); ++i)
                {
                    if (lights[i].expired())
                    {
                        lights.erase(lights.begin() + i);
                        lights.shrink_to_fit();
                    }
                }
            }

            objects.erase(objects.begin() + i);
            objects.shrink_to_fit();
        }
    }
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

bool Scene::isObjectAlreadySampled(unsigned id)
{
    return std::ranges::any_of(lights.begin(), lights.end(), [&](const std::weak_ptr<Object>& object)
    {
        return !object.expired() && object.lock()->object_id == id;
    });
}

void Scene::refreshScene()
{
    need_to_build_intersection_accelerator = true;
}

void Scene::loadSceneFromProject(const ProjectInfo& project_info)
{
    objects.clear();
    lights.clear();

    for (const auto& object_info : project_info.objects_infos)
    {
        auto material = AssetManager::findMaterialAsset(object_info.material_name);
        auto model = AssetManager::findModelAsset(object_info.model_name);

        std::shared_ptr<Object> object;
        if (material->cuda_material->isEmittingLight())
        {
            object = std::make_shared<PointLight>(this, material, model, object_info.position, object_info.rotation, object_info.scale,
              glm::vec3{1, 1, 1}, glm::vec3{1, 0.01, 0.0001});
            lights.push_back(object);
        }
        else
        {
    		object = std::make_shared<Object>(this, material, model, object_info.position, object_info.rotation, object_info.scale);
        }

        object->createShapesForRayTracedMesh();
		objects.push_back(object);
    }

    need_to_build_intersection_accelerator = true;
}

void Scene::createObject(std::shared_ptr<RawModel> model)
{
    const auto new_object = std::make_shared<Object>(this, AssetManager::findMaterialAsset("white"), std::move(model), glm::vec3{0}, glm::vec3{0}, 1.f);
    new_object->createShapesForRayTracedMesh();
    objects.push_back(new_object);
    need_to_build_intersection_accelerator = true;
}

void Scene::buildSceneIntersectionAccelerator()
{
    ShapesCollector shapes_collector{objects};
    const CollectedShapes collected_shapes = shapes_collector.collectShapes();
    scene_light_sources = dmm::DeviceMemoryPointer<Shape*>{collected_shapes.number_of_light_emitting_shapes};
    scene_light_sources.copyFrom(collected_shapes.light_emitting_shapes.data());
    bvh_tree_builder = std::make_unique<BVHTreeBuilder>(collected_shapes);
    intersection_accelerator_tree_traverser = bvh_tree_builder->buildAccelerator();
}

std::vector<ObjectInfo> Scene::gatherObjectsInfos()
{
    std::vector<ObjectInfo> objects_infos;
    objects_infos.reserve(objects.size());
    for (const auto& object : objects)
    {
        objects_infos.push_back(object->getObjectInfo());
    }
    return objects_infos;
}