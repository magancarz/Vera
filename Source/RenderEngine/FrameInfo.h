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
    Camera* camera;
    bool player_moved{false};
    std::map<int, std::shared_ptr<Object>> objects;
    VkImage swap_chain_image;
};