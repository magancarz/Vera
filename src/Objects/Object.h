#pragma once

#include <glm/glm.hpp>

#include "Models/RawModel.h"
#include "RenderEngine/RayTracing/Shapes/Triangle.h"
#include "Objects/ObjectInfo.h"
#include "Utils/DeviceMemoryPointer.h"

struct ShapeInfo;
class MeshMaterial;
class Material;
class Scene;
struct MaterialAsset;
class RayTracedObject;
class Triangle;

class Object
{
public:
    Object(
        Scene* parent_scene,
        std::shared_ptr<MaterialAsset> material,
        std::shared_ptr<RawModel> model_data,
        const glm::vec3& position,
        const glm::vec3& rotation,
        float scale);

    void changeMaterial(std::shared_ptr<MaterialAsset> new_material);
    void refreshObject();

    void increasePosition(float dx, float dy, float dz);
    void increaseRotation(float rx, float ry, float rz);
    glm::vec3 getPosition() const;
    glm::vec3 getRotation() const;
    float getScale() const;
    void setPosition(const glm::vec3& value);
    void setRotation(const glm::vec3& value);
    void setScale(float value);
    void setShouldBeOutlined(bool value);
    bool shouldBeOutlined() const;

    Material* getMaterial() const;
    std::shared_ptr<RawModel> getModelData() const;
    Shape** getTriangles();
    int getNumberOfTriangles();
    ObjectInfo getObjectInfo();


    bool operator==(unsigned int id) const;
    unsigned int object_id;
    std::string name;

    bool isEmittingSomeLight() const { return is_emitting_some_light; }
    size_t getNumberOfTrianglesEmittingLight() const { return triangles_emitting_light.size(); }
    std::vector<Shape*> getTrianglesEmittingLight() const;

    Scene* parent_scene;
    dmm::DeviceMemoryPointer<ShapeInfo> cuda_triangles_infos;

private:
    void createNameForObject();

    void createMeshConsistingOfTriangles();
    void allocateShapesOnDeviceMemory();
    void refreshMeshTrianglesTransforms();
    void changeMeshTrianglesMaterial();
    void gatherShapesEmittingLight();
    glm::mat4 createObjectToWorldTransform();
    glm::mat4 createWorldToObjectTransform();

    std::shared_ptr<MaterialAsset> material;
    std::shared_ptr<RawModel> model_data;

    glm::vec3 position;
    glm::vec3 rotation;
    float scale;
    bool is_emitting_some_light{false};

    std::vector<Shape*> triangles_emitting_light;

    size_t num_of_triangles{0};
    dmm::DeviceMemoryPointer<glm::mat4> object_to_world{};
    dmm::DeviceMemoryPointer<glm::mat4> world_to_object{};
    dmm::DeviceMemoryPointer<Shape*> cuda_shapes;

    bool should_be_outlined = false;

    inline static dmm::DeviceMemoryPointer<size_t> next_triangle_id{};
    inline static size_t next_id = 0;
};
