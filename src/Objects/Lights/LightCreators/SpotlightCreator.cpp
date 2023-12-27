#include "SpotlightCreator.h"

SpotlightCreator::SpotlightCreator()
    : LightCreator{"Spotlight"} {}

bool SpotlightCreator::apply(const std::string& light_info)
{
    return isPrefixValid(light_info, Spotlight::SPOTLIGHT_TAG);
}

std::shared_ptr<Light> SpotlightCreator::fromLightInfo(Scene* parent_scene, const std::string& light_info)
{
    std::stringstream iss(light_info);
    std::string light_info_metadata, tag;
    iss >> light_info_metadata;
    iss >> tag;

    std::string position_x, position_y, position_z,
        light_color_r, light_color_g, light_color_b,
        direction_x, direction_y, direction_z,
        attenuation_x, attenuation_y, attenuation_z,
        cutoff_angle, cutoff_angle_outer;
    iss >> position_x;
    iss >> position_y;
    iss >> position_z;
    glm::vec3 position{std::stof(position_x), std::stof(position_y), std::stof(position_z)};

    iss >> light_color_r;
    iss >> light_color_g;
    iss >> light_color_b;
    glm::vec3 light_color{std::stof(light_color_r), std::stof(light_color_g), std::stof(light_color_b)};

    iss >> direction_x;
    iss >> direction_y;
    iss >> direction_z;
    glm::vec3 direction{std::stof(direction_x), std::stof(direction_y), std::stof(direction_z)};

    iss >> attenuation_x;
    iss >> attenuation_y;
    iss >> attenuation_z;
    glm::vec3 attenuation{std::stof(attenuation_x), std::stof(attenuation_y), std::stof(attenuation_z)};

    iss >> cutoff_angle;
    const float fcutoff_angle = std::stof(cutoff_angle);
    iss >> cutoff_angle_outer;
    const float fcutoff_angle_outer = std::stof(cutoff_angle_outer);

    return std::make_shared<Spotlight>(parent_scene, position, direction, light_color, attenuation, fcutoff_angle, fcutoff_angle_outer);
}

std::shared_ptr<Light> SpotlightCreator::create(Scene* parent_scene)
{
    return std::make_shared<Spotlight>(parent_scene);
}