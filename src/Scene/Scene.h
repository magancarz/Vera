#pragma once

#include <memory>
#include <vector>

#include "Objects/ObjectInfo.h"
#include "Utils/DeviceMemoryPointer.h"
#include "RenderEngine/RayTracing/IntersectionAccelerators/BVHTreeBuilder.h"

class BVHTreeBuilder;
class Triangle;
class BVHTreeTraverser;
struct ProjectInfo;
class BVHAccel;
struct ShapeInfo;
struct RawModel;
class Ray;
class SceneObjectFactory;
class Light;
class Object;
struct MaterialAsset;

class Scene
{
public:
    void loadSceneFromProject(const ProjectInfo& project_info);
    void buildSceneIntersectionAcceleratorIfNeeded();
    void notifyOnObjectChange();
    void notifyOnObjectMaterialChange(const Object* object);
    void addObject(const std::shared_ptr<Object>& scene_object);
    void deleteObject(const Object* scene_object);
    std::weak_ptr<Object> findObjectByID(unsigned int id);
    void refreshScene();
    std::vector<ObjectInfo> gatherObjectsInfos();
    void createObject(std::shared_ptr<RawModel> model);
    void createObject(std::shared_ptr<RawModel> model, std::shared_ptr<MaterialAsset> material, glm::vec3 position, glm::vec3 rotation, float scale);

    std::vector<std::shared_ptr<Object>> objects;
    std::vector<std::weak_ptr<Object>> lights;
    dmm::DeviceMemoryPointer<BVHTreeTraverser> intersection_accelerator_tree_traverser;
    dmm::DeviceMemoryPointer<Shape*> scene_light_sources;

private:
    void buildSceneIntersectionAccelerator();
    bool isObjectAlreadySampled(unsigned int id);
    
    bool need_to_build_intersection_accelerator{false};
};
