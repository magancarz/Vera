#include "WindowSystem.h"

void WindowSystem::initialize(std::unique_ptr<Window> window)
{
    window_impl = std::move(window);
}
