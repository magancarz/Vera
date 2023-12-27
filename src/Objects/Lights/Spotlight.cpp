#include "Spotlight.h"

Spotlight::Spotlight(Scene* parent_scene, const glm::vec3& position, const glm::vec4& light_direction, const glm::vec3& light_color,
        const glm::vec3& attenuation, float cutoff_angle, float cutoff_angle_outer)
    : Light(parent_scene, position, light_direction, light_color, attenuation, cutoff_angle, cutoff_angle_outer) {}

std::string Spotlight::getObjectInfo()
{
    return SPOTLIGHT_TAG + " " + Light::getObjectInfo() + " " + Algorithms::vec4ToString(light_direction) + " " + Algorithms::vec3ToString(attenuation) + " " +
           Algorithms::floatToString(cutoff_angle_cosine) + " " + Algorithms::floatToString(cutoff_angle_offset_cosine);
}