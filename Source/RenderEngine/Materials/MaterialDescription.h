#pragma once

struct MaterialDescription
{
    uint32_t material_hit_group_index;
    VkDeviceAddress material_info_buffer_device_address;
    VkImageLayout material_texture_layout;
    VkImageView material_texture_view;
    VkSampler material_texture_sampler;
};