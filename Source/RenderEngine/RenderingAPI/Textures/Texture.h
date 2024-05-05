#pragma once

#include <string>

#include "RenderEngine/RenderingAPI/Device.h"

class Texture
{
public:
    Texture(Device& device, const std::string& filepath);
    Texture(Device& device, VkFormat image_format, uint32_t width, uint32_t height);
    ~Texture();

    VkSampler getSampler() { return sampler; }
    VkImageView getImageView() { return image_view; }
    VkImageLayout getImageLayout() { return image_layout; }

private:
    void createTexture();
    void transitionImageLayout(VkImageLayout old_layout, VkImageLayout new_layout);
    void generateMipmaps();

    uint32_t width, height;
    uint32_t mip_levels;

    Device& device;
    VkImage image;
    VkDeviceMemory image_memory;
    VkImageView image_view;
    VkSampler sampler;
    VkFormat image_format;
    VkImageLayout image_layout;
};
