#pragma once

#include "Objects/Lights/Light.h"

class DirectionalLight : public Light
{
public:
    DirectionalLight(Scene* parent_scene);
    DirectionalLight(
        Scene* parent_scene,
        const glm::vec3& light_direction,
        const glm::vec3& light_color);
    ~DirectionalLight() override = default;

    std::string getObjectInfo() override;
    void renderObjectInformationGUI() override;

    inline static std::string DIRECTIONAL_LIGHT_TAG{"directional_light"};

protected:
    void createShadowMapTexture() override;
    void createLightSpaceTransform() override;
};