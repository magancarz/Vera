#pragma once

#include "Light.h"
#include "RenderEngine/Shadows/CubeShadowMapShader.h"

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

    void loadTransformationMatrixToShadowMapShader(const glm::mat4& mat) override;

    void setPosition(const glm::vec3& value) override;

    int getLightType() const override;
    std::string getObjectInfo() override;
    void renderObjectInformationGUI() override;

    inline static std::string POINT_LIGHT_TAG{"point_light"};

protected:
    void createShadowMapShader() override;
    void createShadowMapTexture() override;
    void createLightSpaceTransform() override;

    std::vector<glm::mat4> light_view_transforms{6};

private:
    CubeShadowMapShader* shader_program_as_cube_shadow_map_shader{nullptr};
};