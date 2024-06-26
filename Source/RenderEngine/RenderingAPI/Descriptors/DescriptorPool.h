#pragma once

#include <memory>

#include "RenderEngine/RenderingAPI/VulkanHandler.h"

class DescriptorPool
{
public:
    DescriptorPool(
            VulkanHandler& device,
            uint32_t max_sets,
            VkDescriptorPoolCreateFlags pool_flags,
            const std::vector<VkDescriptorPoolSize>& pool_sizes);
    ~DescriptorPool();

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool &operator=(const DescriptorPool&) = delete;
    DescriptorPool(DescriptorPool&&) = delete;
    DescriptorPool &operator=(DescriptorPool&&) = delete;

    VkDescriptorPool descriptorPool() { return descriptor_pool; }

    bool allocateDescriptor(VkDescriptorSetLayout descriptor_set_layout, VkDescriptorSet& descriptor) const;
    void freeDescriptors(std::vector<VkDescriptorSet>& descriptors) const;
    void resetPool();

private:
    VulkanHandler& device;
    VkDescriptorPool descriptor_pool;

    friend class DescriptorWriter;
};