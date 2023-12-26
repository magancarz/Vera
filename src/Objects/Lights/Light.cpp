#include "Light.h"

Light::Light(Scene* parent_scene, std::shared_ptr<MaterialAsset> material, std::shared_ptr<RawModel> model_data,
        const glm::vec3& position, const glm::vec3& rotation, float scale, const glm::vec4& light_direction,
        const glm::vec3& light_color, const glm::vec3& attenuation, float cutoff_angle_cosine)
    : Object(parent_scene, std::move(material), std::move(model_data), position, rotation, scale),
      light_direction(light_direction), light_color(light_color), attenuation(attenuation), cutoff_angle_cosine(cutoff_angle_cosine) {}

glm::vec4 Light::getLightDirection() const
{
    return light_direction;
}

void Light::setLightDirection(const glm::vec4& in_light_direction)
{
    light_direction = in_light_direction;
}

glm::vec3 Light::getLightColor() const
{
    return light_color;
}

void Light::setLightColor(const glm::vec3& in_light_color)
{
    light_color = in_light_color;
}

float Light::getCutoffAngle() const
{
    return cutoff_angle_cosine;
}

void Light::setCutoffAngle(float in_cutoff_angle_cosine)
{
    cutoff_angle_cosine = in_cutoff_angle_cosine;
}

float Light::getCutoffAngleOffset() const
{
    return cutoff_angle_offset_cosine;
}

void Light::setCutoffAngleOffset(float in_cutoff_angle_offset_cosine)
{
    cutoff_angle_offset_cosine = in_cutoff_angle_offset_cosine;
}

glm::vec3 Light::getAttenuation() const
{
    return attenuation;
}

void Light::setAttenuation(const glm::vec3& in_attenuation)
{
    attenuation = in_attenuation;
}