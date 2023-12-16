#pragma once

#include "glm/glm.hpp"
#include <glm/gtc/type_ptr.hpp>

#define CAMERA_HEIGHT 5

enum direction { FORWARD, BACKWARD, LEFT, RIGHT };

class Camera
{
public:
    Camera(const glm::vec3& position);

    bool move();
    void updateCameraVectors();

    void invertPitch();

    void createProjectionMatrix();
    glm::mat4 getProjectionMatrix() { return projection_matrix; }

    glm::mat4 getView() const;

    void setPosition(const glm::vec3& position);
    void setPitch(float value);
    void setYaw(float yaw);
    void setRoll(float roll);

    void increasePitch(float pitch);
    void increaseYaw(float yaw);
    void increaseRoll(float roll);

    glm::vec3 getPosition() const;
    glm::vec3 getDirection() const { return position + front; }
    glm::vec3 getWorldUpVector() const { return world_up; }
    float getFieldOfView() const { return field_of_view; }
    float getPitch() const;
    float getYaw() const;
    float getRoll() const;
    float getSensitivity() const;
    glm::vec3 getCameraFront() const;
    glm::vec3 getCameraRight() const;

private:
    void setFront(const glm::vec3& camera_front);
    void setRight(const glm::vec3& camera_right);

    void checkInputs();

    inline static const float RUN_SPEED = 10.0f;
    inline static const float TURN_SPEED = 160.0f;

    float forward_speed = 0;
    float upwards_speed = 0;
    float sideways_speed = 0;
    float pitch_change = 0;
    float yaw_change = 0;
    float field_of_view = 70.f;

    glm::vec3 position;
    glm::vec3 world_up = glm::vec3(0, 1, 0);
    glm::vec3 front = glm::vec3(0, 0, 1);
    glm::vec3 right = glm::vec3(1, 0, 0);
    glm::vec3 up = glm::vec3(0, 1, 0);

    const float FOV = 70.f;
    const float NEAR_PLANE = 0.1f;
    const float FAR_PLANE = 1000.f;

    float movement_speed, sensitivity;
    float pitch, yaw, roll;

    glm::mat4 projection_matrix;
};
