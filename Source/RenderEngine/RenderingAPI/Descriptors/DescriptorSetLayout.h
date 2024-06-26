#pragma once

#include <memory>
#include <unordered_map>

#include "RenderEngine/RenderingAPI/VulkanHandler.h"

class DescriptorSetLayout
{
public:
    DescriptorSetLayout(VulkanHandler& device, std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings);
    ~DescriptorSetLayout();

    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout &operator=(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout(DescriptorSetLayout&&) = delete;
    DescriptorSetLayout &operator=(DescriptorSetLayout&&) = delete;

    VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptor_set_layout; }

private:
    VulkanHandler& device;
    VkDescriptorSetLayout descriptor_set_layout;
    std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings;

    friend class DescriptorWriter;
};