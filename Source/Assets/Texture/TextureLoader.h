#pragma once

#include "Memory/MemoryAllocator.h"
#include "TextureData.h"

class DeviceTexture;

class TextureLoader
{
public:
    static std::unique_ptr<DeviceTexture> loadFromAssetFile(
        VulkanFacade& vulkan_facade,
        MemoryAllocator& memory_allocator,
        const std::string& texture_name,
        VkFormat format);

private:
    static TextureData loadTextureData(const std::string& texture_resource_location);
    static std::unique_ptr<DeviceTexture> createTextureOnDevice(
        VulkanFacade& vulkan_facade,
        MemoryAllocator& memory_allocator,
        const TextureData& texture_data,
        VkFormat format);
};
