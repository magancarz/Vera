#pragma once

#include "glm/glm.hpp"
#include "glm/glm/gtc/type_ptr.hpp"

class Camera
{
public:
    Camera(const glm::vec3& position);

    bool move();

    glm::mat4 getPerspectiveProjectionMatrix() const;
    glm::mat4 getCameraViewMatrix() const;

    glm::vec3 getPosition() const;
    glm::vec3 getCameraFrontVector() const { return position + front; }
    glm::vec3 getCameraUpVector() const { return up; }
    float getFieldOfView() const { return FOV; }

private:
    void checkInputs();
    void updateCameraDirectionVectors();

    float forward_speed{0};
    float upwards_speed{0};
    float sideways_speed{0};
    float pitch_change{0};
    float yaw_change{0};

    glm::vec3 position{0, 0, 0};
    glm::vec3 world_up{0, 1, 0};
    glm::vec3 front{0, 0, 1};
    glm::vec3 right{1, 0, 0};
    glm::vec3 up{0, 1, 0};

    const float FOV{70.f};
    const float NEAR_PLANE{.1f};
    const float FAR_PLANE{1000.f};

    float movement_speed{10.f};
    float mouse_sensitivity{.35f};
    float pitch{0};
    float yaw{0};

    glm::mat4 perspective_projection_matrix{};
    glm::mat4 orthographic_projection_matrix{};
};
