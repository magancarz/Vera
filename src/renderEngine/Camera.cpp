#include "Camera.h"

#include "input/Input.h"
#include "GUI/Display.h"

Camera::Camera(const glm::vec3& position)
    : position(position) {}

bool Camera::move()
{
    bool did_camera_move = false;

    checkInputs();
    updateCameraDirectionVectors();

    if (pitch >= 90.f)
        pitch = 89.999f;
    else if (pitch <= -90.f)
        pitch = -89.999f;

    if (yaw >= 360.f || yaw <= -360.f)
        yaw = 0.f;

    if (forward_speed != 0 || sideways_speed != 0 || upwards_speed != 0 || pitch_change != 0 || yaw_change != 0)
        did_camera_move = true;

    position += front * forward_speed * static_cast<float>(Display::getFrameTimeSeconds());
    position += right * sideways_speed * static_cast<float>(Display::getFrameTimeSeconds());
    position += up * upwards_speed * static_cast<float>(Display::getFrameTimeSeconds());

    pitch += pitch_change;
    yaw += yaw_change;

    pitch_change = 0;
    yaw_change = 0;

    return did_camera_move;
}

void Camera::checkInputs()
{
    if (Input::isKeyDown(GLFW_KEY_W))
    {
        forward_speed = movement_speed;
    }
    else if (Input::isKeyDown(GLFW_KEY_S))
    {
        forward_speed = -movement_speed;
    }
    else
    {
        forward_speed = 0;
    }

    if (Input::isKeyDown(GLFW_KEY_A))
    {
        sideways_speed = -movement_speed;
    }
    else if (Input::isKeyDown(GLFW_KEY_D))
    {
        sideways_speed = movement_speed;
    }
    else
    {
        sideways_speed = 0;
    }

    if (Input::isKeyDown(GLFW_KEY_E))
    {
        upwards_speed = movement_speed;
    }
    else if (Input::isKeyDown(GLFW_KEY_Q))
    {
        upwards_speed = -movement_speed;
    }
    else
    {
        upwards_speed = 0;
    }

    if (Input::isRightMouseButtonDown())
    {
        Display::disableCursor();

        pitch_change = static_cast<float>(Display::getMouseYOffset() * mouse_sensitivity);
        yaw_change = static_cast<float>(Display::getMouseXOffset() * mouse_sensitivity);
    }
    else
    {
        Display::enableCursor();
    }
}

void Camera::updateCameraDirectionVectors()
{
    front.x = cos(glm::radians(yaw - 90.f)) * cos(glm::radians(-pitch));
    front.y = sin(glm::radians(-pitch));
    front.z = sin(glm::radians(yaw - 90.f)) * cos(glm::radians(-pitch));

    front = normalize(front);
    right = normalize(cross(front, up));
}

glm::mat4 Camera::getCameraViewMatrix() const
{
    return glm::lookAt(position, position + front, up);
}

glm::mat4 Camera::getPerspectiveProjectionMatrix() const
{
    return glm::perspective(glm::radians(FOV), Display::ASPECT, NEAR_PLANE, FAR_PLANE);
}

glm::vec3 Camera::getPosition() const
{
    return position;
}
