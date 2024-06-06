#pragma once

#include "Memory/MemoryAllocator.h"
#include "TextureData.h"

class DeviceTexture;

class TextureLoader
{
public:
    static std::unique_ptr<DeviceTexture> loadFromAssetFile(
        VulkanHandler& vulkan_facade,
        MemoryAllocator& memory_allocator,
        const std::string& texture_name,
        VkFormat format);

private:
    static TextureData loadTextureData(const std::string& texture_resource_location);
    static std::unique_ptr<DeviceTexture> createTextureOnDevice(
        VulkanHandler& vulkan_facade,
        MemoryAllocator& memory_allocator,
        const TextureData& texture_data,
        VkFormat format);
};
