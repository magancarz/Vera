#include "GLFWInputManager.h"

#include "GLFWKeyMappings.h"
#include "KeyMappingsSystem.h"
#include "Editor/Window/GLFWWindow.h"
#include "Editor/Window/WindowSystem.h"
#include "Logs/LogSeverity.h"
#include "Logs/LogSystem.h"

GLFWInputManager::GLFWInputManager()
{
    obtainGLFWwindowPointer();
    KeyMappingsSystem::initialize(std::make_unique<GLFWKeyMappings>());
}

void GLFWInputManager::obtainGLFWwindowPointer()
{
    if (const auto as_glfw_window = dynamic_cast<GLFWWindow*>(&WindowSystem::get()))
    {
        window = as_glfw_window->getGFLWwindow();
    }
    else
    {
        LogSystem::log(LogSeverity::FATAL, "Couldn't obtain glfw window pointer!");
    }
}

bool GLFWInputManager::isKeyPressed(KeyCode key_mapping)
{
    return glfwGetKey(window, KeyMappingsSystem::getImplKeyCodeFor(key_mapping)) == GLFW_PRESS;
}