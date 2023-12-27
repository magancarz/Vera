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
struct ShapeInfo;
struct RawModel;
class Ray;
class Light;
class Object;
class TriangleMesh;
struct MaterialAsset;

class Scene
{
public:
    void loadSceneFromProject(const ProjectInfo& project_info);
    void buildSceneIntersectionAcceleratorIfNeeded();
    void notifyOnObjectChange();
    void notifyOnObjectMaterialChange();
    void deleteObject(const Object* scene_object);
    std::weak_ptr<Object> findObjectByID(unsigned int id);
    void refreshScene();
    std::vector<std::string> gatherObjectsInfos();
    void createObject(std::shared_ptr<RawModel> model);

    std::vector<std::shared_ptr<Object>> objects;
    std::vector<std::shared_ptr<TriangleMesh>> triangle_meshes;
    std::vector<std::shared_ptr<Light>> lights;

    dmm::DeviceMemoryPointer<BVHTreeTraverser> intersection_accelerator_tree_traverser;
    dmm::DeviceMemoryPointer<Shape*> scene_light_sources;

private:
    void buildSceneIntersectionAccelerator();

    std::unique_ptr<BVHTreeBuilder> bvh_tree_builder;
    bool need_to_build_intersection_accelerator{false};
};
