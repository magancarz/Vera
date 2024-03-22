#pragma once

#include <string>

#include "RenderEngine/RenderingAPI/Device.h"

class Texture
{
public:
    Texture(Device& device, const std::string& filepath);
    ~Texture();

    VkSampler getSampler() { return sampler; }
    VkImageView getImageView() { return image_view; }
    VkImageLayout getImageLayout() { return image_layout; }

private:
    void transitionImageLayout(VkImageLayout old_layout, VkImageLayout new_layout);
    void generateMipmaps();

    int width, height;
    uint32_t mip_levels;

    Device& device;
    VkImage image;
    VkDeviceMemory image_memory;
    VkImageView image_view;
    VkSampler sampler;
    VkFormat image_format;
    VkImageLayout image_layout;
};
