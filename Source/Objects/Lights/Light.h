#pragma once

#include "Objects/Object.h"

class Light : public Object
{
public:
    Light(
        Scene* parent_scene,
        const glm::vec3& position = {0, 0, 0},
        const glm::vec3& light_direction = {0, -1, 0},
        const glm::vec3& light_color = {1, 1, 1},
        const glm::vec3& attenuation = {1, 0, 0},
        float cutoff_angle_cosine = 0.f,
        float cutoff_angle_outer_cosine = 0.f);

    ~Light() override = default;

    std::string getObjectInfo() override;
    void renderObjectInformationGUI() override;

    bool shouldBeOutlined() const override;

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

    glm::vec3 light_direction{0, -1, 0};
    glm::vec3 light_color{1, 1, 1};
    glm::vec3 attenuation{0, 0, 0};
    float cutoff_angle_cosine{0};
    float cutoff_angle_offset_cosine{0.85f};
};