#pragma once

#include "Object.h"

namespace cudaObjectUtils
{
    __device__ void transformShape(
        glm::mat4* object_to_world, glm::mat4* world_to_object,
        Shape* shape, ShapeInfo* shapes_info);
}

class TriangleMesh : public Object
{
public:
    TriangleMesh(
        Scene* parent_scene,
        std::shared_ptr<MaterialAsset> material,
        std::shared_ptr<RawModel> model_data,
        const glm::vec3& position,
        const glm::vec3& rotation,
        float scale);
    ~TriangleMesh() override;

    std::string getObjectInfo() override;

    void changeMaterial(std::shared_ptr<MaterialAsset> new_material);
    void createShapesForRayTracedMesh();
    void renderObjectInformationGUI() override;

    void setPosition(const glm::vec3 &value) override;
    glm::vec3 getRotation() const;
    float getScale() const;
    void setRotation(const glm::vec3& value);
    void setScale(float value);

    Material* getMaterial() const;
    std::shared_ptr<RawModel> getModelData() const;
    Shape** getShapes();
    ShapeInfo* getShapesInfos();
    size_t getNumberOfShapes();

    size_t getNumberOfLightEmittingShapes() const { return shapes_emitting_light.size(); }
    std::vector<Shape*> getShapesEmittingLight() const;

protected:
    void createNameForObject();
    void refreshObject();
    void resetMeshTrianglesTransforms();
    void transformMeshTriangles();
    void changeMeshShapesMaterial();
    void gatherShapesEmittingLight();
    virtual void createShapesOnDeviceMemory();
    virtual bool determineIfShapeIsEmittingLight(size_t i);
    glm::mat4 createObjectToWorldTransform();
    glm::mat4 createWorldToObjectTransform();

    std::shared_ptr<MaterialAsset> material{};
    std::shared_ptr<RawModel> model_data{};

    glm::vec3 rotation{};
    float scale{};
    bool is_emitting_some_light{false};

    std::vector<Shape*> shapes_emitting_light{};

    size_t num_of_shapes{0};
    dmm::DeviceMemoryPointer<glm::mat4> object_to_world{};
    dmm::DeviceMemoryPointer<glm::mat4> world_to_object{};
    dmm::DeviceMemoryPointer<Shape*> shapes{};
    dmm::DeviceMemoryPointer<ShapeInfo> shapes_infos;

    inline static dmm::DeviceMemoryPointer<size_t> next_shape_id{};
    inline static size_t next_id = 0;
};
