#include "PointLight.h"

PointLight::PointLight(Scene* parent_scene, const glm::vec3& position, const glm::vec3& light_color, const glm::vec3& attenuation)
    : Light(parent_scene, position, {0, -1, 0, 1}, light_color, attenuation) {}

std::string PointLight::getObjectInfo()
{
    return POINT_LIGHT_TAG + " " + Light::getObjectInfo() + " " + Algorithms::vec3ToString(attenuation);
}