#pragma once

#include <string>

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "TextureData.h"

class Texture
{
public:
    Texture(VulkanFacade& device, const std::string& filepath);
    Texture(VulkanFacade& device, uint32_t width, uint32_t height, VkImageUsageFlags usage_flags, VkFormat image_format = VK_FORMAT_R8G8B8A8_SRGB);
    ~Texture();

    VkSampler getSampler() { return sampler; }
    VkImageView getImageView() { return image_view; }
    VkImageLayout getImageLayout() { return image_layout; }

private:
    VulkanFacade& device;

    void createImage(VkImageUsageFlags usage_flags);
    void createImageView();
    void createImageSampler();

    void copyDataToImage(const TextureData& texture_data);
    void transitionImageLayout(VkImageLayout old_layout, VkImageLayout new_layout);
    void generateMipmaps();

    uint32_t width, height;
    VkFormat image_format;
    uint32_t mip_levels{1};

    VkImage image{VK_NULL_HANDLE};
    VkImageView image_view{VK_NULL_HANDLE};
    VkSampler sampler{VK_NULL_HANDLE};

    VkDeviceMemory image_memory{VK_NULL_HANDLE};
    VkImageLayout image_layout{VK_IMAGE_LAYOUT_UNDEFINED};
};
