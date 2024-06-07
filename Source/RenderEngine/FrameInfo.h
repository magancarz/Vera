#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>

struct FrameInfo
{
    VkCommandBuffer command_buffer;
    VkExtent2D window_size;
    float delta_time{0};

    glm::mat4 camera_view_matrix{1.f};
    glm::mat4 camera_projection_matrix{1.f};

    VkDescriptorSet ray_traced_texture;

    glm::vec3 sun_position{glm::normalize(glm::vec3{1})};
    float weather{0.05f};
    bool need_to_refresh_generated_image{false};
};