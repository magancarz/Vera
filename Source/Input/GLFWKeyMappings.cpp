#include "GLFWKeyMappings.h"

#include "GLFW/glfw3.h"

GLFWKeyMappings::GLFWKeyMappings()
{
    key_mappings.emplace(MOVE_LEFT, GLFW_KEY_A);
    key_mappings.emplace(MOVE_RIGHT, GLFW_KEY_D);
    key_mappings.emplace(MOVE_FORWARD, GLFW_KEY_W);
    key_mappings.emplace(MOVE_BACKWARD, GLFW_KEY_S);
    key_mappings.emplace(MOVE_UP, GLFW_KEY_E);
    key_mappings.emplace(MOVE_DOWN, GLFW_KEY_Q);
    key_mappings.emplace(LOOK_LEFT, GLFW_KEY_LEFT);
    key_mappings.emplace(LOOK_RIGHT, GLFW_KEY_RIGHT);
    key_mappings.emplace(LOOK_UP, GLFW_KEY_UP);
    key_mappings.emplace(LOOK_DOWN, GLFW_KEY_DOWN);
}
