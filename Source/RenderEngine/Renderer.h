#pragma once

#include <vulkan/vulkan.hpp>

#include "Camera.h"

class Renderer
{
public:
    Renderer();

    void renderScene();

private:
    VkDevice vk_device;
};
