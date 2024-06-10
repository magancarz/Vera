#pragma once

struct QueueFamilyIndices
{
    uint32_t graphicsFamily;
    uint32_t presentFamily;
    bool graphics_family_has_value = false;
    bool present_family_has_value = false;

    bool isComplete() const
    {
        return graphics_family_has_value && present_family_has_value;
    }
};