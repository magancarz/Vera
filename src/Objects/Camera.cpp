#include "Camera.h"

#include "Input/Input.h"
#include "GUI/Display.h"

Camera::Camera(const glm::vec3& position)
    : position(position)
{
    movement_speed = 0.3f;
    sensitivity = 0.35f;

    pitch = 35.0f;
    yaw = 0.0f;
    roll = 0.0f;

    createProjectionMatrix();
}

void Camera::invertPitch()
{
    pitch = -pitch;
}

glm::mat4 Camera::getView() const
{
    auto view_matrix = glm::mat4(1.0f);

    view_matrix = rotate(view_matrix, glm::radians(pitch), glm::vec3(1, 0, 0));
    view_matrix = rotate(view_matrix, glm::radians(yaw), glm::vec3(0, 1, 0));

    const auto camera_pos = getPosition();
    const auto negative_camera_pos = glm::vec3(-camera_pos[0], -camera_pos[1], -camera_pos[2]);

    view_matrix = translate(view_matrix, negative_camera_pos);

    return view_matrix;
}

bool Camera::move()
{
    bool out_value = false;

    checkInputs();
    updateCameraVectors();

    if (pitch > 89.999f)
        pitch = 89.999f;
    else if (pitch < -89.999f)
        pitch = -89.999f;

    if (yaw > 360.f || yaw < -360.f)
        yaw = 0.f;

    if (forward_speed != 0.0f || sideways_speed != 0.0f || upwards_speed != 0.0f || pitch_change != 0.0f || yaw_change != 0.0f)
        out_value = true;

    position += front * forward_speed * static_cast<float>(Display::getFrameTimeSeconds());
    position += right * sideways_speed * static_cast<float>(Display::getFrameTimeSeconds());
    position += up * upwards_speed * static_cast<float>(Display::getFrameTimeSeconds());

    pitch += pitch_change;
    yaw += yaw_change;

    pitch_change = 0;
    yaw_change = 0;

    return out_value;
}

void Camera::setFront(const glm::vec3& camera_front)
{
    front = camera_front;
}

void Camera::setRight(const glm::vec3& camera_right)
{
    right = camera_right;
}

void Camera::checkInputs()
{
    if (Input::isKeyDown(GLFW_KEY_W))
    {
        forward_speed = RUN_SPEED;
    }
    else if (Input::isKeyDown(GLFW_KEY_S))
    {
        forward_speed = -RUN_SPEED;
    }
    else
    {
        forward_speed = 0;
    }

    if (Input::isKeyDown(GLFW_KEY_A))
    {
        sideways_speed = -RUN_SPEED;
    }
    else if (Input::isKeyDown(GLFW_KEY_D))
    {
        sideways_speed = RUN_SPEED;
    }
    else
    {
        sideways_speed = 0;
    }

    if (Input::isKeyDown(GLFW_KEY_E))
    {
        upwards_speed = RUN_SPEED;
    }
    else if (Input::isKeyDown(GLFW_KEY_Q))
    {
        upwards_speed = -RUN_SPEED;
    }
    else
    {
        upwards_speed = 0;
    }

    if (Input::isRightMouseButtonDown())
    {
        Display::disableCursor();

        pitch_change = static_cast<float>(Display::getMouseYOffset() * 0.2);
        yaw_change = static_cast<float>(Display::getMouseXOffset() * 0.2);
    }
    else
    {
        Display::enableCursor();
    }
}

void Camera::updateCameraVectors()
{
    front.x = cos(glm::radians(yaw - 90.f)) * cos(glm::radians(-pitch));
    front.y = sin(glm::radians(-pitch));
    front.z = sin(glm::radians(yaw - 90.f)) * cos(glm::radians(-pitch));

    front = normalize(front);
    right = normalize(cross(front, up));
}

void Camera::createProjectionMatrix()
{
    projection_matrix = glm::mat4();
    constexpr float aspect_ratio = static_cast<float>(Display::WINDOW_WIDTH) / static_cast<float>(
        Display::WINDOW_HEIGHT);
    const float y_scale = 1.0f / glm::tan(glm::radians(FOV / 2.0f));
    const float x_scale = y_scale / aspect_ratio;
    const float frustum_length = FAR_PLANE - NEAR_PLANE;

    projection_matrix[0][0] = x_scale;
    projection_matrix[1][1] = y_scale;
    projection_matrix[2][2] = -((FAR_PLANE + NEAR_PLANE) / frustum_length);
    projection_matrix[2][3] = -1;
    projection_matrix[3][2] = -((2.0f * NEAR_PLANE * FAR_PLANE) / frustum_length);
    projection_matrix[3][3] = 0;
}

void Camera::setPosition(const glm::vec3& pos)
{
    position = pos;
}

void Camera::setPitch(const float value)
{
    pitch = value;
}

void Camera::setYaw(const float value)
{
    yaw = value;
}

void Camera::setRoll(const float value)
{
    roll = value;
}

void Camera::increasePitch(const float value)
{
    pitch += value * sensitivity;
}

void Camera::increaseYaw(const float value)
{
    yaw += value * sensitivity;
}

void Camera::increaseRoll(const float value)
{
    roll += value * sensitivity;
}

glm::vec3 Camera::getPosition() const
{
    return position;
}

float Camera::getPitch() const
{
    return pitch;
}

float Camera::getYaw() const
{
    return yaw;
}

float Camera::getRoll() const
{
    return roll;
}

float Camera::getSensitivity() const
{
    return sensitivity;
}

glm::vec3 Camera::getCameraFront() const
{
    return front;
}

glm::vec3 Camera::getCameraRight() const
{
    return right;
}
