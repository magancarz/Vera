#include "DirectionalLightCreator.h"

DirectionalLightCreator::DirectionalLightCreator()
    : LightCreator{"Directional Light"} {}

bool DirectionalLightCreator::apply(const std::string& light_info)
{
    return isPrefixValid(light_info, DirectionalLight::DIRECTIONAL_LIGHT_TAG);
}

std::shared_ptr<Light> DirectionalLightCreator::fromLightInfo(Scene* parent_scene, const std::string& light_info)
{
    std::stringstream iss(light_info);
    std::string light_info_metadata, tag;
    iss >> light_info_metadata;
    iss >> tag;

    std::string position_x, position_y, position_z,
        light_color_r, light_color_g, light_color_b,
        direction_x, direction_y, direction_z;
    iss >> position_x;
    iss >> position_y;
    iss >> position_z;

    iss >> light_color_r;
    iss >> light_color_g;
    iss >> light_color_b;
    glm::vec3 light_color{std::stof(light_color_r), std::stof(light_color_g), std::stof(light_color_b)};

    iss >> direction_x;
    iss >> direction_y;
    iss >> direction_z;
    glm::vec3 direction{std::stof(direction_x), std::stof(direction_y), std::stof(direction_z)};

    auto light = std::make_shared<DirectionalLight>(parent_scene, direction, light_color);
    light->prepare();
    return light;
}

std::shared_ptr<Light> DirectionalLightCreator::create(Scene* parent_scene)
{
    auto light = std::make_shared<DirectionalLight>(parent_scene);
    light->prepare();
    return light;
}