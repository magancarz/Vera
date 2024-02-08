#include "Camera.h"

#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>

Camera::Camera(const glm::vec3& position)
    : position(position)
{
    setViewDirection(position, {0, 0, 1});
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
    return glm::perspective(glm::radians(FOV), static_cast<float>(1280.f/800), NEAR_PLANE, FAR_PLANE);
}

glm::vec3 Camera::getPosition() const
{
    return position;
}
