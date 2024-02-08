#include "Camera.h"

#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>

#include "input/Input.h"
#include "GUI/Display.h"

Camera::Camera(const glm::vec3& position)
    : position(position)
{
    setViewDirection(position, {0, 0, 1});
}

bool Camera::move()
{
    bool did_camera_move = false;

    checkInputs();
    updateCameraDirectionVectors();

    if (pitch > 89.999f)
        pitch = 89.999f;
    else if (pitch < -89.999f)
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

void Camera::setViewDirection(const glm::vec3& position, const glm::vec3& direction, const glm::vec3& up)
{
    const glm::vec3 w{glm::normalize(direction)};
    const glm::vec3 u{glm::normalize(glm::cross(w, up))};
    const glm::vec3 v{glm::cross(w, u)};

    view_matrix = glm::mat4{1.f};
    view_matrix[0][0] = u.x;
    view_matrix[1][0] = u.y;
    view_matrix[2][0] = u.z;
    view_matrix[0][1] = v.x;
    view_matrix[1][1] = v.y;
    view_matrix[2][1] = v.z;
    view_matrix[0][2] = w.x;
    view_matrix[1][2] = w.y;
    view_matrix[2][2] = w.z;
    view_matrix[3][0] = -glm::dot(u, position);
    view_matrix[3][1] = -glm::dot(v, position);
    view_matrix[3][2] = -glm::dot(w, position);
}

glm::mat4 Camera::getViewMatrix() const
{
    return view_matrix;
}

glm::mat4 Camera::getPerspectiveProjectionMatrix() const
{
    return glm::perspective(glm::radians(FOV), Display::getAspect(), NEAR_PLANE, FAR_PLANE);
}

glm::vec3 Camera::getPosition() const
{
    return position;
}
