#include "Scene.h"

#include "editor/Project/Project.h"
#include "Models/AssetManager.h"
#include "RenderEngine/RayTracing/PDF/HittablePDF.h"
#include "Objects/Object.h"
#include "RenderEngine/RayTracing/IntersectionAccelerators/BVHTreeBuilder.h"
#include "RenderEngine/RayTracing/Shapes/ShapeInfo.h"

Scene::Scene()
{
    intersection_accelerator_builder = std::make_shared<BVHTreeBuilder>();
}

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

void Scene::addObject(const std::shared_ptr<Object>& scene_object)
{
    objects.push_back(scene_object);
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
        
		auto object = std::make_shared<Object>(this, material, model, object_info.position, object_info.rotation, object_info.scale);
		objects.push_back(object);

        if (object->isEmittingSomeLight())
        {
            lights.push_back(object);
        }
    }

    need_to_build_intersection_accelerator = true;
}

void Scene::createObject(std::shared_ptr<RawModel> model)
{
    objects.push_back(std::make_shared<Object>(this, AssetManager::findMaterialAsset("white"), std::move(model), glm::vec3{0}, glm::vec3{0}, 1.f));
    need_to_build_intersection_accelerator = true;
}

void Scene::createObject(std::shared_ptr<RawModel> model, std::shared_ptr<MaterialAsset> material, glm::vec3 position, glm::vec3 rotation, float scale)
{
    const auto new_object = std::make_shared<Object>(this, std::move(material), std::move(model), position, rotation, scale);
    objects.push_back(new_object);
    if (new_object->isEmittingSomeLight())
    {
        lights.push_back(new_object);
    }
    need_to_build_intersection_accelerator = true;
}

void Scene::buildSceneIntersectionAccelerator()
{
    gatherShapesFromScene();
    const std::vector<ShapeInfo*> shapes_infos = getShapesInfos();
    dmm::DeviceMemoryPointer<ShapeInfo*> temp1(shapes_infos.size());
    temp1.copyFrom(shapes_infos.data());
    intersection_accelerator_tree_traverser = intersection_accelerator_builder->buildAccelerator(cuda_scene_triangles, temp1);
}

void Scene::gatherShapesFromScene()
{
    size_t num_of_triangles_in_scene = 0;
    size_t num_of_light_sources = 0;
    for (const auto& object : objects)
    {
        const int object_triangles = object->getNumberOfTriangles();
        num_of_triangles_in_scene += object_triangles;
        if (object->isEmittingSomeLight())
        {
            num_of_light_sources += object->getNumberOfTrianglesEmittingLight();
        }
    }

    std::vector<Triangle*> temp_light_sources;
    temp_light_sources.reserve(num_of_light_sources);
    for (auto& object : lights)
    {
        if (object.expired()) continue;

        std::vector<Triangle*> object_light_sources = object.lock()->getTrianglesEmittingLight();
        for (size_t i = 0; i < object_light_sources.size(); ++i)
        {
			temp_light_sources.push_back(object_light_sources[i]);
        }
    }
    scene_light_sources = dmm::DeviceMemoryPointer<Triangle*>(num_of_light_sources);
    scene_light_sources.copyFrom(temp_light_sources.data());

    std::vector<Triangle*> temp_cuda_scene_triangles;
    temp_cuda_scene_triangles.reserve(num_of_triangles_in_scene);
    for (const auto& object : objects)
    {
        Triangle* temp = object->getTriangles();
        for (int i = 0; i < object->getNumberOfTriangles(); ++i)
        {
            temp_cuda_scene_triangles.push_back(&temp[i]);
        }
    }
    cuda_scene_triangles = dmm::DeviceMemoryPointer<Triangle*>(num_of_triangles_in_scene);
    cuda_scene_triangles.copyFrom(temp_cuda_scene_triangles.data());
}

std::vector<ShapeInfo*> Scene::getShapesInfos()
{
    std::vector<ShapeInfo*> shapes_infos;
    shapes_infos.reserve(cuda_scene_triangles.size());
    for (const auto& object : objects)
    {
        ShapeInfo* temp = object->cuda_triangles_infos.data();
        for (int i = 0; i < object->getNumberOfTriangles(); ++i)
        {
            shapes_infos.push_back(&temp[i]);
        }
    }

    return shapes_infos;
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