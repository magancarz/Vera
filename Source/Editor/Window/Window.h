#pragma once

#include <vulkan/vulkan_core.h>

#include "Utils/VeraDefines.h"

class Window
{
public:
    explicit Window(
            uint32_t width = constants::DEFAULT_WINDOW_WIDTH,
            uint32_t height = constants::DEFAULT_WINDOW_HEIGHT,
            std::string name = constants::DEFAULT_WINDOW_TITLE);
    virtual ~Window() = default;

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    [[nodiscard]] virtual bool shouldClose() const = 0;
    [[nodiscard]] VkExtent2D getExtent() const { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}; }
    [[nodiscard]] float getAspect() const { return static_cast<float>(width) / static_cast<float>(height); }
    [[nodiscard]] bool wasWindowResized() const { return window_resized; }
    void resetWindowResizedFlag() { window_resized = false; }

protected:
    uint32_t width;
    uint32_t height;
    std::string window_name;

    bool window_resized = false;
};
