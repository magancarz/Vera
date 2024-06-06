#include <cmath>
#include "DeviceTexture.h"

#include "Memory/Buffer.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "Memory/Image.h"

DeviceTexture::DeviceTexture(
        VulkanHandler& vulkan_facade,
        MemoryAllocator& memory_allocator,
        const TextureData& texture_data,
        const VkImageCreateInfo& image_info,
        std::unique_ptr<Image> image)
    : vulkan_facade{vulkan_facade}, name{texture_data.name}, channels{texture_data.number_of_channels}, image_info{image_info}, image_buffer{std::move(image)}
{
    checkIfTextureIsOpaque(texture_data.data);
    copyDataToImage(memory_allocator, texture_data.data);
    generateMipmaps();
    createImageSampler();
    createImageView();
}

void DeviceTexture::checkIfTextureIsOpaque(const std::vector<unsigned char>& texture_data)
{
    is_opaque = channels == 3 || texture_data[3] > 0.5;
}

void DeviceTexture::copyDataToImage(MemoryAllocator& memory_allocator, const std::vector<unsigned char>& texture_data)
{
    transitionImageLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    auto buffer = memory_allocator.createStagingBuffer(4, image_info.extent.width * image_info.extent.height, texture_data.data());
    copyBufferToImage(buffer->getBuffer());
}

void DeviceTexture::copyBufferToImage(VkBuffer src_buffer)
{
    VkCommandBuffer command_buffer = vulkan_facade.getCommandPool().beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {image_info.extent.width, image_info.extent.height, 1};

    vkCmdCopyBufferToImage(
            command_buffer,
            src_buffer,
            image_buffer->getImage(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region);
    vulkan_facade.getCommandPool().endSingleTimeCommands(command_buffer);
}

void DeviceTexture::createImageView()
{
    VkImageViewCreateInfo image_view_create_info{};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_create_info.format = image_info.format;
    image_view_create_info.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
    image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_create_info.subresourceRange.baseMipLevel = 0;
    image_view_create_info.subresourceRange.baseArrayLayer = 0;
    image_view_create_info.subresourceRange.layerCount = 1;
    image_view_create_info.subresourceRange.levelCount = image_info.mipLevels;
    image_view_create_info.image = image_buffer->getImage();

    if (vkCreateImageView(vulkan_facade.getDeviceHandle(), &image_view_create_info, VulkanDefines::NO_CALLBACK, &image_view) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create image view!");
    }
}

void DeviceTexture::createImageSampler()
{
    VkSamplerCreateInfo sampler_create_info{};
    sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_create_info.magFilter = VK_FILTER_LINEAR;
    sampler_create_info.minFilter = VK_FILTER_LINEAR;
    sampler_create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.mipLodBias = 0.0f;
    sampler_create_info.compareOp = VK_COMPARE_OP_NEVER;
    sampler_create_info.minLod = 0.0f;
    sampler_create_info.maxLod = static_cast<float>(image_info.mipLevels);
    sampler_create_info.maxAnisotropy = 4.0;
    sampler_create_info.anisotropyEnable = VK_TRUE;
    sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

    if (vkCreateSampler(vulkan_facade.getDeviceHandle(), &sampler_create_info, VulkanDefines::NO_CALLBACK, &sampler) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create image sampler!");
    }
}

DeviceTexture::DeviceTexture(VulkanHandler& vulkan_facade, std::unique_ptr<Image> image, const VkImageCreateInfo& image_info)
    : vulkan_facade{vulkan_facade}, image_info{image_info}, image_buffer{std::move(image)}
{
    createImageView();
    createImageSampler();
    transitionImageLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
}

DeviceTexture::~DeviceTexture()
{
    vkDestroyImageView(vulkan_facade.getDeviceHandle(), image_view, VulkanDefines::NO_CALLBACK);
    vkDestroySampler(vulkan_facade.getDeviceHandle(), sampler, VulkanDefines::NO_CALLBACK);
}

void DeviceTexture::transitionImageLayout(VkImageLayout old_layout, VkImageLayout new_layout)
{
    VkCommandBuffer command_buffer = vulkan_facade.getCommandPool().beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image_buffer->getImage();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = image_info.mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags source_stage;
    VkPipelineStageFlags destination_stage;

    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_GENERAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = 0;

        source_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        destination_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    else
    {
        throw std::runtime_error("Unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
            command_buffer,
            source_stage,
            destination_stage,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier);

    vulkan_facade.getCommandPool().endSingleTimeCommands(command_buffer);

    image_layout = new_layout;
}

void DeviceTexture::generateMipmaps()
{
    VkFormatProperties format_properties;
    vkGetPhysicalDeviceFormatProperties(vulkan_facade.getPhysicalDeviceHandle(), image_info.format, &format_properties);

    if (!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
    {
        throw std::runtime_error("Texture image format does not support linear blitting!");
    }

    VkCommandBuffer command_buffer = vulkan_facade.getCommandPool().beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image_buffer->getImage();
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mip_width = image_info.extent.width;
    int32_t mip_height = image_info.extent.height;

    for (uint32_t i = 1; i < image_info.mipLevels; ++i)
    {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(
                command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mip_width, mip_height, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mip_width > 1 ? mip_width / 2 : 1, mip_height > 1 ? mip_height / 2 : 1, 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(
                command_buffer,
                image_buffer->getImage(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image_buffer->getImage(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blit,
                VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
                command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &barrier);

        if (mip_width > 1) mip_width /= 2;
        if (mip_height > 1) mip_height /= 2;
    }

    barrier.subresourceRange.baseMipLevel = image_info.mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier);

    vulkan_facade.getCommandPool().endSingleTimeCommands(command_buffer);

    image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}