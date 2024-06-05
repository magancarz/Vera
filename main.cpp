#include "Vera.h"

#include <iostream>
#include <Logs/BasicLogger.h>
#include <Logs/LogSystem.h>

#include "Editor/Window/GLFWWindow.h"
#include "Editor/Window/WindowSystem.h"
#include "Source/Assets/Model/Vertex.h"

int main()
{
    LogSystem::initialize(std::make_unique<BasicLogger>());
    WindowSystem::initialize(std::make_unique<GLFWWindow>());

    Vera app{};

    try
    {
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
