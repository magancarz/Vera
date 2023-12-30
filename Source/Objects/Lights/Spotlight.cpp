#include "Spotlight.h"
#include "GUI/GUI.h"

Spotlight::Spotlight(Scene* parent_scene)
    : Light(parent_scene, {0, 0, 0}, {0, -1, 0}, {1, 1, 1}, {1, 0.01, 0.0001}, 0.9f, 0.85f) {}

Spotlight::Spotlight(Scene* parent_scene, const glm::vec3& position, const glm::vec3& light_direction, const glm::vec3& light_color,
        const glm::vec3& attenuation, float cutoff_angle, float cutoff_angle_outer)
    : Light(parent_scene, position, light_direction, light_color, attenuation, cutoff_angle, cutoff_angle_outer) {}

std::string Spotlight::getObjectInfo()
{
    return SPOTLIGHT_TAG + " " + Light::getObjectInfo() + " " + Algorithms::vec3ToString(light_direction) + " " + Algorithms::vec3ToString(attenuation) + " " +
           Algorithms::floatToString(cutoff_angle_cosine) + " " + Algorithms::floatToString(cutoff_angle_offset_cosine);
}

void Spotlight::renderObjectInformationGUI()
{
    Light::renderObjectInformationGUI();

    auto light_direction_value = GUI::drawInputFieldForVector3(light_direction, "Light Direction");
    if (light_direction_value.has_value())
    {
        setLightDirection(light_direction_value.value());
    }

    auto attenuation_value = GUI::drawInputFieldForVector3(attenuation, "Attenuation");
    if (attenuation_value.has_value())
    {
        setAttenuation(attenuation_value.value());
    }

    GUI::drawInputFieldForFloat(&cutoff_angle_cosine, "Cutoff Angle Cosine");
    GUI::drawInputFieldForFloat(&cutoff_angle_offset_cosine, "Cutoff Angle Offset Cosine");
}