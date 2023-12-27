#include "PointLightCreator.h"

PointLightCreator::PointLightCreator()
    : LightCreator{"Point Light"} {}

bool PointLightCreator::apply(const std::string& light_info)
{
    return isPrefixValid(light_info, PointLight::POINT_LIGHT_TAG);
}

std::shared_ptr<Light> PointLightCreator::fromLightInfo(Scene* parent_scene, const std::string& light_info)
{
    std::stringstream iss(light_info);
    std::string light_info_metadata, tag;
    iss >> light_info_metadata;
    iss >> tag;

    std::string position_x, position_y, position_z,
        light_color_r, light_color_g, light_color_b,
        attenuation_x, attenuation_y, attenuation_z;
    iss >> position_x;
    iss >> position_y;
    iss >> position_z;
    glm::vec3 position{std::stof(position_x), std::stof(position_y), std::stof(position_z)};

    iss >> light_color_r;
    iss >> light_color_g;
    iss >> light_color_b;
    glm::vec3 light_color{std::stof(light_color_r), std::stof(light_color_g), std::stof(light_color_b)};

    iss >> attenuation_x;
    iss >> attenuation_y;
    iss >> attenuation_z;
    glm::vec3 attenuation{std::stof(attenuation_x), std::stof(attenuation_y), std::stof(attenuation_z)};

    return std::make_shared<PointLight>(parent_scene, position, light_color, attenuation);
}

std::shared_ptr<Light> PointLightCreator::create(Scene* parent_scene)
{
    return std::make_shared<PointLight>(parent_scene);
}