#include "DirectionalLight.h"
#include "GUI/GUI.h"

DirectionalLight::DirectionalLight(Scene* parent_scene)
    : Light(parent_scene) {}

DirectionalLight::DirectionalLight(Scene* parent_scene, const glm::vec4& light_direction, const glm::vec3& light_color)
    : Light(parent_scene, {0, 0, 0}, light_direction, light_color) {}

std::string DirectionalLight::getObjectInfo()
{
    return DIRECTIONAL_LIGHT_TAG + " " + Light::getObjectInfo() + " " + Algorithms::vec4ToString(light_direction);
}

void DirectionalLight::renderObjectInformationGUI()
{
    Light::renderObjectInformationGUI();

    auto light_direction_value = GUI::drawInputFieldForVector4(light_direction, "Light Direction");
    if (light_direction_value.has_value())
    {
        setLightDirection(light_direction_value.value());
    }
}