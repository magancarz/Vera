#pragma once

#include "RenderEngine/Camera.h"
#include "Objects/Object.h"

#include <vulkan/vulkan.hpp>
#include <map>

struct FrameInfo
{
    VkCommandBuffer command_buffer;
    VkExtent2D window_size;
    float frame_time;

    Camera* camera;

    VkDescriptorSet ray_traced_texture;

    glm::vec3 sun_position{glm::normalize(glm::vec3{1})};
    float weather{0.05f};
    bool need_to_refresh_generated_image{false};
};