#pragma once

#include "Light.h"

class PointLight : public Light
{
public:
    PointLight(Scene* parent_scene);
    PointLight(
        Scene* parent_scene,
        const glm::vec3& position,
        const glm::vec3& light_color,
        const glm::vec3& attenuation);

    ~PointLight() override = default;

    std::string getObjectInfo() override;
    void renderObjectInformationGUI() override;

    inline static std::string POINT_LIGHT_TAG{"point_light"};
};