#pragma once

#include "RenderEngine/Camera.h"
#include "Objects/Object.h"

#include <vulkan/vulkan.hpp>
#include <map>

struct FrameInfo
{
    VkCommandBuffer command_buffer;
    float frame_time;
    Camera* camera;
    bool player_moved{false};
    VkDescriptorSet ray_traced_texture;
};