#pragma once

#include "DescriptorPool.h"
#include "RenderEngine/RenderingAPI/VulkanHandler.h"

class DescriptorPoolBuilder
{
public:
    explicit DescriptorPoolBuilder(VulkanHandler& device) : device{device} {}

    DescriptorPoolBuilder& addPoolSize(VkDescriptorType descriptor_type, uint32_t count);
    DescriptorPoolBuilder& setPoolFlags(VkDescriptorPoolCreateFlags flags);
    DescriptorPoolBuilder& setMaxSets(uint32_t count);
    [[nodiscard]] std::unique_ptr<DescriptorPool> build() const;

private:
    VulkanHandler& device;
    std::vector<VkDescriptorPoolSize> pool_sizes{};
    uint32_t max_sets{1000};
    VkDescriptorPoolCreateFlags pool_flags{0};
};
