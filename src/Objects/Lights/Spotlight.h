#pragma once

#include "Objects/Lights/Light.h"

class Spotlight : public Light
{
public:
    explicit Spotlight(Scene* parent_scene);
    Spotlight(
        Scene* parent_scene,
        const glm::vec3& position,
        const glm::vec3& light_direction,
        const glm::vec3& light_color,
        const glm::vec3& attenuation,
        float cutoff_angle, float cutoff_angle_outer);
    ~Spotlight() override = default;

    std::string getObjectInfo() override;
    void renderObjectInformationGUI() override;

    inline static std::string SPOTLIGHT_TAG{"spotlight"};
};