#pragma once

#include <string>

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "TextureData.h"
#include "Memory/MemoryAllocator.h"

class Texture
{
public:
    Texture(VulkanFacade& device, MemoryAllocator& memory_allocator, const std::string& texture_name, VkFormat image_format = VK_FORMAT_R8G8B8A8_SRGB);
    Texture(VulkanFacade& device, MemoryAllocator& memory_allocator, uint32_t width, uint32_t height, VkImageUsageFlags usage_flags, VkFormat image_format = VK_FORMAT_R8G8B8A8_SRGB);
    ~Texture();

    std::string getName() const { return name; }

    VkSampler getSampler() const { return sampler; }
    VkImageView getImageView() const { return image_view; }
    VkImageLayout getImageLayout() const { return image_layout; }

    [[nodiscard]] bool isOpaque() const { return is_opaque; }

private:
    VulkanFacade& device;
    MemoryAllocator& memory_allocator;

    std::string name;

    void checkIfTextureIsOpaque(const TextureData& texture_data);
    void createImage(VkImageUsageFlags usage_flags);
    void createImageView();
    void createImageSampler();

    void copyDataToImage(const TextureData& texture_data);
    void transitionImageLayout(VkImageLayout old_layout, VkImageLayout new_layout);
    void generateMipmaps();

    uint32_t width, height;
    VkFormat image_format;
    uint32_t mip_levels{1};
    bool is_opaque{true};

    VkImage image{VK_NULL_HANDLE};
    VkImageView image_view{VK_NULL_HANDLE};
    VkSampler sampler{VK_NULL_HANDLE};

    VkDeviceMemory image_memory{VK_NULL_HANDLE};
    VkImageLayout image_layout{VK_IMAGE_LAYOUT_UNDEFINED};
};
