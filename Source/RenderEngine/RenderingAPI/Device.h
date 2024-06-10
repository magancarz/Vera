#pragma once

#include "Instance.h"
#include "PhysicalDevice.h"

class Device
{
public:
    Device(const Instance& instance, const PhysicalDevice& physical_device);
    ~Device();

    [[nodiscard]] VkDevice getDevice() const { return device; }
    [[nodiscard]] VkQueue getGraphicsQueue() const { return graphics_queue; }
    [[nodiscard]] VkQueue getPresentQueue() const { return present_queue; }

private:
    void createLogicalDevice(const Instance& instance, const PhysicalDevice& physical_device);

    VkDevice device;
    VkQueue graphics_queue;
    VkQueue present_queue;
};
