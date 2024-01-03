#pragma once

#include "Objects/Object.h"
#include "RenderEngine/Shadows/ShadowMapShader.h"

class Light : public Object
{
public:
    Light(
        Scene* parent_scene,
        const glm::vec3& position = {0, 0, 0},
        const glm::vec3& light_direction = {0, -1, 0},
        const glm::vec3& light_color = {1, 1, 1},
        const glm::vec3& attenuation = {1.f, 0.01f, 0.0001f},
        float cutoff_angle_cosine = 0.f,
        float cutoff_angle_outer_cosine = 0.f);

    ~Light() override = default;

    virtual void prepare();

    void prepareForShadowMapRendering();
    virtual void bindShadowMapTexture() const;
    virtual void bindShadowMapTextureToFramebuffer() const;
    glm::mat4 getLightSpaceTransform() const;
    virtual void loadTransformationMatrixToShadowMapShader(const glm::mat4& mat) = 0;
    std::shared_ptr<utils::Texture> getShadowMap() const;

    std::string getObjectInfo() override;
    void renderObjectInformationGUI() override;

    virtual int getLightType() const = 0;

    bool shouldBeOutlined() const override;

    void setPosition(const glm::vec3& value) override;
    glm::vec3 getLightDirection() const;
    void setLightDirection(const glm::vec3& in_light_direction);
    glm::vec3 getLightColor() const;
    void setLightColor(const glm::vec3& in_light_color);
    float getCutoffAngle() const;
    void setCutoffAngle(float in_cutoff_angle_cosine);
    float getCutoffAngleOffset() const;
    void setCutoffAngleOffset(float in_cutoff_angle_offset_cosine);
    glm::vec3 getAttenuation() const;
    void setAttenuation(const glm::vec3& in_attenuation);

protected:
    void createNameForLight();
    virtual void createShadowMapShader() = 0;
    virtual void createShadowMapTexture();
    virtual void createLightSpaceTransform() = 0;

    unsigned int shadow_map_width = 1024;
    unsigned int shadow_map_height = 1024;
    float shadow_map_orthographic_projection_x_span = 10.f;
    float shadow_map_orthographic_projection_y_span = 10.f;
    float near_plane = 1.0f;
    float far_plane = 30.f;

    std::unique_ptr<ShaderProgram> shadow_map_shader;
    std::shared_ptr<utils::Texture> shadow_map_texture;
    glm::mat4 light_space_transform;

    glm::vec3 light_direction{0, -1, 0};
    glm::vec3 light_color{1, 1, 1};
    glm::vec3 attenuation{0, 0, 0};
    float cutoff_angle_cosine{0};
    float cutoff_angle_offset_cosine{0.85f};
};