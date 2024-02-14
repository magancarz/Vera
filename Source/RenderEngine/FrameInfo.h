#pragma once

#include "RenderEngine/Camera.h"

#include <vulkan/vulkan.hpp>

struct FrameInfo
{
    int frame_index;
    float frame_time;
    VkCommandBuffer command_buffer;
    Camera& camera;
    VkDescriptorSet global_descriptor_set;
};