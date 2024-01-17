#pragma once

#include <memory>
#include <vector>
#include <map>

#include "Objects/TriangleMeshInfo.h"
#include "Utils/DeviceMemoryPointer.h"
#include "RenderEngine/RayTracing/IntersectionAccelerators/BVHTreeBuilder.h"
#include "Objects/Lights/LightCreators/LightCreator.h"
#include "Objects/Lights/LightCreators/DirectionalLightCreator.h"
#include "Objects/Lights/LightCreators/PointLightCreator.h"
#include "Objects/Lights/LightCreators/SpotlightCreator.h"

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
    void deleteObject(Object* scene_object);
    std::weak_ptr<Object> findObjectByID(unsigned int id);
    void refreshScene();
    void createTriangleMesh(std::shared_ptr<RawModel> model);
    void createSceneLight(const std::shared_ptr<LightCreator>& light_creator);

    std::vector<std::string> gatherTriangleMeshesInfos();
    std::vector<std::string> gatherLightsInfos();

    std::vector<std::shared_ptr<Object>> objects;
    std::vector<std::weak_ptr<TriangleMesh>> triangle_meshes;
    std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>> triangle_meshes_map;
    std::vector<std::weak_ptr<Light>> lights;
    dmm::DeviceMemoryPointer<BVHTreeTraverser> intersection_accelerator_tree_traverser;
    dmm::DeviceMemoryPointer<Shape*> scene_light_sources;

private:
    void buildSceneIntersectionAccelerator();
    void loadTriangleMeshes(const std::vector<std::string>& triangle_meshes_infos);
    void loadLights(const std::vector<std::string>& lights_infos);
    void putTriangleMeshToMapOfModels(const std::weak_ptr<TriangleMesh>& triangle_mesh);
    void removeTriangleMeshFromMapOfModels(const TriangleMesh* triangle_mesh);

    std::unique_ptr<BVHTreeBuilder> bvh_tree_builder;
    bool need_to_build_intersection_accelerator{false};
};
