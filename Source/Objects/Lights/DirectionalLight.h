#pragma once

#include "Objects/Lights/Light.h"

class DirectionalLight : public Light
{
public:
    explicit DirectionalLight(Scene* parent_scene);
    DirectionalLight(
        Scene* parent_scene,
        const glm::vec3& light_direction,
        const glm::vec3& light_color);
    ~DirectionalLight() override = default;

    void loadTransformationMatrixToShadowMapShader(const glm::mat4& mat) override;

    int getLightType() const override;
    std::string getObjectInfo() override;
    void renderObjectInformationGUI() override;

    inline static std::string DIRECTIONAL_LIGHT_TAG{"directional_light"};

protected:
    void createShadowMapShader() override;
    void createLightSpaceTransform() override;

private:
    ShadowMapShader* shader_program_as_shadow_map_shader{nullptr};
};