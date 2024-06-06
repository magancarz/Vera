#include "TextureLoader.h"

#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "Assets/Defines.h"
#include "Logs/LogSystem.h"
#include "Utils/PathBuilder.h"
#include "RenderEngine/Textures/DeviceTexture.h"
#include "Utils/Algorithms.h"

std::unique_ptr<DeviceTexture> TextureLoader::loadFromAssetFile(
        VulkanHandler& vulkan_facade,
        MemoryAllocator& memory_allocator,
        const std::string& texture_name,
        VkFormat format)
{
    TextureData texture_data = loadTextureData(texture_name);
    return createTextureOnDevice(vulkan_facade, memory_allocator, texture_data, format);
}

TextureData TextureLoader::loadTextureData(const std::string& texture_resource_location)
{
    const std::string location = PathBuilder(Assets::TEXTURES_DIRECTORY_PATH).append(texture_resource_location).build();

    int width, height, number_of_channels;
    constexpr int EXPECTED_NUMBER_OF_CHANNELS = 4;
    unsigned char* data = stbi_load(
        location.c_str(),
        &width,
        &height,
        &number_of_channels,
        EXPECTED_NUMBER_OF_CHANNELS);

    if (!data)
    {
        LogSystem::log(LogSeverity::FATAL, "Failed to load image ", texture_resource_location);
        throw std::runtime_error("Failed to load image " + texture_resource_location);
    }

    if (number_of_channels != EXPECTED_NUMBER_OF_CHANNELS)
    {
        LogSystem::log(LogSeverity::ERROR, "Encountered different number of channels than expected with texture ", texture_resource_location, "!");
    }

    std::vector<unsigned char> copied_data(width * height * EXPECTED_NUMBER_OF_CHANNELS);
    memcpy(copied_data.data(), data, copied_data.size() * sizeof(unsigned char));
    stbi_image_free(data);

    return
    {
        .name = texture_resource_location,
        .width = static_cast<uint32_t>(width),
        .height = static_cast<uint32_t>(height),
        .number_of_channels = static_cast<uint32_t>(EXPECTED_NUMBER_OF_CHANNELS),
        .data = std::move(copied_data)
    };
}

std::unique_ptr<DeviceTexture> TextureLoader::createTextureOnDevice(
        VulkanHandler& vulkan_facade,
        MemoryAllocator& memory_allocator,
        const TextureData& texture_data,
        VkFormat format)
{
    VkImageCreateInfo image_create_info{};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.extent.width = texture_data.width;
    image_create_info.extent.height = texture_data.height;
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texture_data.width, texture_data.height)))) + 1;
    image_create_info.arrayLayers = 1;
    image_create_info.format = format;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;

    std::unique_ptr<Image> image = memory_allocator.createImage(image_create_info);
    return std::make_unique<DeviceTexture>(
        vulkan_facade,
        memory_allocator,
        texture_data,
        image_create_info,
        std::move(image));
}
