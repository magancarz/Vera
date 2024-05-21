#include "Surface.h"
#include "RenderEngine/Window.h"
#include "VulkanDefines.h"

Surface::Surface(Instance& instance)
    : instance{instance}
{
    createSurface();
}

Surface::~Surface()
{
    vkDestroySurfaceKHR(instance.getInstance(), surface, VulkanDefines::NO_CALLBACK);
}

void Surface::createSurface()
{
    auto window = Window::get();
    window->createWindowSurface(instance.getInstance(), &surface);
}