#pragma once

#include <string>
#include <Assets/Texture/TextureData.h>

#include "RenderEngine/RenderingAPI/VulkanHandler.h"
#include "Memory/MemoryAllocator.h"

class DeviceTexture
{
public:
    DeviceTexture(
        VulkanHandler& vulkan_facade,
        MemoryAllocator& memory_allocator,
        const TextureData& texture_info,
        std::unique_ptr<Image> image_buffer);
    DeviceTexture(VulkanHandler& vulkan_facade, const TextureData& texture_data, std::unique_ptr<Image> image_buffer);
    ~DeviceTexture();

    DeviceTexture(const DeviceTexture&) = delete;
    DeviceTexture &operator=(const DeviceTexture&) = delete;
    DeviceTexture(DeviceTexture&&) = delete;
    DeviceTexture &operator=(DeviceTexture&&) = delete;

    [[nodiscard]] std::string getName() const { return texture_info.name; }

    [[nodiscard]] VkSampler getSampler() const { return sampler; }
    [[nodiscard]] VkImageView getImageView() const { return image_view; }
    [[nodiscard]] VkImageLayout getImageLayout() const { return image_layout; }

    [[nodiscard]] VkDescriptorImageInfo descriptorInfo() const;

    [[nodiscard]] bool isOpaque() const { return texture_info.is_opaque; }

private:
    VulkanHandler& vulkan_facade;
    TextureData texture_info;
    std::unique_ptr<Image> image_buffer{};

    void createImageView();
    void createImageSampler();

    void copyDataToImage(MemoryAllocator& memory_allocator, const std::vector<unsigned char>& texture_data);
    void copyBufferToImage(VkBuffer buffer);
    void transitionImageLayout(VkImageLayout old_layout, VkImageLayout new_layout);
    void generateMipmaps();

    VkImageView image_view{VK_NULL_HANDLE};
    VkSampler sampler{VK_NULL_HANDLE};

    VkImageLayout image_layout{VK_IMAGE_LAYOUT_UNDEFINED};
};
