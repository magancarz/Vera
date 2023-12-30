#include "Object.h"

#include "Materials/MaterialAsset.h"
#include "Scene/Scene.h"
#include "Materials/Material.h"
#include "renderEngine/RayTracing/Shapes/ShapeInfo.h"

Object::Object(
    Scene* parent_scene,
    const glm::vec3& position)
    : parent_scene(parent_scene),
      position(position) {}

glm::vec3 Object::getPosition() const
{
    return position;
}

void Object::setPosition(const glm::vec3& value)
{
    position = value;
}

bool Object::operator==(size_t id) const
{
    return object_id == id;
}

void Object::setShouldBeOutlined(bool value)
{
    should_be_outlined = value;
}

bool Object::shouldBeOutlined() const
{
    return should_be_outlined;
}
