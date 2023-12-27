#include "DirectionalLight.h"

DirectionalLight::DirectionalLight(Scene* parent_scene, const glm::vec3& position, const glm::vec4& light_direction, const glm::vec3& light_color)
    : Light(parent_scene, position, light_direction, light_color) {}

std::string DirectionalLight::getObjectInfo()
{
    return DIRECTIONAL_LIGHT_TAG + " " + Light::getObjectInfo() + " " + Algorithms::vec4ToString(light_direction);
}