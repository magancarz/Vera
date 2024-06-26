#pragma once

#include "Instance.h"

class Surface
{
public:
    explicit Surface(Instance& instance);
    ~Surface();

    Surface(const Surface&) = delete;
    Surface &operator=(const Surface&) = delete;
    Surface(Surface&&) = delete;
    Surface &operator=(Surface&&) = delete;

    [[nodiscard]] VkSurfaceKHR getSurface() const { return surface; }

private:
    void createSurface();

    Instance& instance;
    VkSurfaceKHR surface{VK_NULL_HANDLE};
};
