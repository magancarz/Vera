#include "Object.h"

Object Object::createObject()
{
    static id_t available_id = 0;
    return Object{available_id++};
}

Object Object::createPointLight(float intensity, float radius, const glm::vec3& color)
{
    Object object = Object::createObject();
    object.color = color;
    object.transform_component.scale.x = radius;
    object.point_light = std::make_unique<PointLightComponent>();
    object.point_light->light_intensity = intensity;
    return object;
}