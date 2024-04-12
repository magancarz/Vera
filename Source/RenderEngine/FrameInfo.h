#pragma once

#include "RenderEngine/Camera.h"
#include "Objects/Object.h"

#include <vulkan/vulkan.hpp>
#include <map>

struct FrameInfo
{
    int frame_index;
    float frame_time;
    VkCommandBuffer command_buffer;
    VkDescriptorSet global_uniform_buffer_descriptor_set;
    Camera* camera;
    std::map<int, std::shared_ptr<Object>> objects;
    VkImage swap_chain_image;
};