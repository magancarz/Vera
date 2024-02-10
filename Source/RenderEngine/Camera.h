#pragma once

#include "glm/glm.hpp"

class Camera
{
public:
    Camera(const glm::vec3& position);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getPerspectiveProjectionMatrix(float aspect) const;

    void setViewDirection(const glm::vec3& position, const glm::vec3& direction, const glm::vec3& up = {0, -1, 0});
    void setViewTarget(glm::vec3 position, glm::vec3 target, glm::vec3 up = glm::vec3{0.f, -1.f, 0.f});
    void setViewYXZ(glm::vec3 position, glm::vec3 rotation);

private:
    glm::vec3 position{0, 0, 0};
    glm::vec3 world_up{0, 1, 0};
    glm::vec3 front{0, 0, 1};
    glm::vec3 right{1, 0, 0};
    glm::vec3 up{0, -1, 0};

    const float FOV{70.f};
    const float NEAR_PLANE{.1f};
    const float FAR_PLANE{1000.f};

    glm::mat4 perspective_projection_matrix{};
    glm::mat4 orthographic_projection_matrix{};
    glm::mat4 view_matrix{};
};
