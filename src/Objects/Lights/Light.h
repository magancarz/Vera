#pragma once

#include "Objects/Object.h"

class Light : public Object
{
public:
    Light(
        Scene* parent_scene,
        const glm::vec3& position,
        const glm::vec4& light_direction = {0, -1, 0, 1},
        const glm::vec3& light_color = {1, 1, 1},
        const glm::vec3& attenuation = {1, 0, 0},
        float cutoff_angle_cosine = 0.f,
        float cutoff_angle_outer_cosine = 0.f);

    ~Light() override = default;

    std::string getObjectInfo() override;
    void renderObjectInformationGUI() override;

    glm::vec4 getLightDirection() const;
    void setLightDirection(const glm::vec4& in_light_direction);
    glm::vec3 getLightColor() const;
    void setLightColor(const glm::vec3& in_light_color);
    float getCutoffAngle() const;
    void setCutoffAngle(float in_cutoff_angle_cosine);
    float getCutoffAngleOffset() const;
    void setCutoffAngleOffset(float in_cutoff_angle_offset_cosine);
    glm::vec3 getAttenuation() const;
    void setAttenuation(const glm::vec3& in_attenuation);

    bool shouldBeOutlined() const override;

protected:
    glm::vec4 light_direction{0, -1, 0, 1};
    glm::vec3 light_color{1, 1, 1};
    glm::vec3 attenuation{1, 0.01, 0.0001};
    float cutoff_angle_cosine{0};
    float cutoff_angle_offset_cosine{0.85f};
};