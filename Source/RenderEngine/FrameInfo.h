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
    Camera& camera;
    std::vector<VkDescriptorSet> global_descriptor_sets;
    std::map<int, Object>& objects;
};