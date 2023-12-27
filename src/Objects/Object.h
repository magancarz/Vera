#pragma once

#include <glm/glm.hpp>

#include "Models/RawModel.h"
#include "RenderEngine/RayTracing/Shapes/Triangle.h"
#include "Objects/TriangleMeshInfo.h"
#include "Utils/DeviceMemoryPointer.h"

struct ShapeInfo;
class Material;
class Scene;
struct MaterialAsset;
class Triangle;

class Object
{
public:
    Object(
        Scene* parent_scene,
        const glm::vec3& position);

    virtual ~Object() = default;

    glm::vec3 getPosition() const;
    virtual void setPosition(const glm::vec3& value);
    virtual std::string getObjectInfo() = 0;
    bool operator==(size_t id) const;
    void setShouldBeOutlined(bool value);
    virtual bool shouldBeOutlined() const;
    virtual void renderObjectInformationGUI() = 0;

    std::string name;
    size_t object_id;

protected:
    Scene* parent_scene;

    glm::vec3 position{};
    bool should_be_outlined = false;

    inline static size_t next_id = 0;
};
