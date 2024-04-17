#include "KeyboardMovementController.h"

void KeyboardMovementController::moveInPlaneXZ(GLFWwindow* window, float delta_time, Object& object)
{
    glm::vec3 rotate{0.f};
    if (glfwGetKey(window, keys.look_right) == GLFW_PRESS) rotate.y += 1.f;
    if (glfwGetKey(window, keys.look_left) == GLFW_PRESS) rotate.y -= 1.f;
    if (glfwGetKey(window, keys.look_up) == GLFW_PRESS) rotate.x -= 1.f;
    if (glfwGetKey(window, keys.look_down) == GLFW_PRESS) rotate.x += 1.f;

    if (glm::dot(rotate, rotate) > std::numeric_limits<float>::epsilon())
    {
        object.transform_component.rotation += look_speed * delta_time * glm::normalize(rotate);
    }

    player_moved = rotate != glm::vec3{0};

    object.transform_component.rotation.x = glm::clamp(object.transform_component.rotation.x, -1.5f, 1.5f);
    object.transform_component.rotation.y = glm::mod(object.transform_component.rotation.y, glm::two_pi<float>());

    float yaw = object.transform_component.rotation.y;
    const glm::vec3 forward_dir{sin(yaw), 0.f, cos(yaw)};
    const glm::vec3 right_dir{forward_dir.z, 0.f, -forward_dir.x};
    const glm::vec3 up_dir{0.f, -1.f, 0.f};

    glm::vec3 move_dir{0.f};
    if (glfwGetKey(window, keys.move_forward) == GLFW_PRESS) move_dir += forward_dir;
    if (glfwGetKey(window, keys.move_backward) == GLFW_PRESS) move_dir -= forward_dir;
    if (glfwGetKey(window, keys.move_right) == GLFW_PRESS) move_dir += right_dir;
    if (glfwGetKey(window, keys.move_left) == GLFW_PRESS) move_dir -= right_dir;
    if (glfwGetKey(window, keys.move_up) == GLFW_PRESS) move_dir += up_dir;
    if (glfwGetKey(window, keys.move_down) == GLFW_PRESS) move_dir -= up_dir;

    player_moved |= move_dir != glm::vec3{0};

    if (glm::dot(move_dir, move_dir) > std::numeric_limits<float>::epsilon())
    {
        object.transform_component.translation += move_speed * delta_time * glm::normalize(move_dir);
    }
}

bool KeyboardMovementController::playerMoved() const
{
    return player_moved;
}