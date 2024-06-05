#pragma once

#include <string>
#include <Assets/Texture/TextureData.h>

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "Memory/MemoryAllocator.h"

class DeviceTexture
{
public:
    DeviceTexture(
            VulkanFacade& vulkan_facade,
            MemoryAllocator& memory_allocator,
            const TextureData& texture_data,
            const VkImageCreateInfo& image_info,
            std::unique_ptr<Image> image);
    DeviceTexture(
            VulkanFacade& vulkan_facade,
            std::unique_ptr<Image> image, const VkImageCreateInfo& image_info);
    ~DeviceTexture();

    [[nodiscard]] std::string getName() const { return name; }

    [[nodiscard]] VkSampler getSampler() const { return sampler; }
    [[nodiscard]] VkImageView getImageView() const { return image_view; }
    [[nodiscard]] VkImageLayout getImageLayout() const { return image_layout; }

    [[nodiscard]] bool isOpaque() const { return is_opaque; }

private:
    VulkanFacade& vulkan_facade;
    std::string name{};
    bool is_opaque{true};
    uint32_t channels{3};
    VkImageCreateInfo image_info{};
    std::unique_ptr<Image> image_buffer{};

    void checkIfTextureIsOpaque(const std::vector<unsigned char>& texture_data);
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
