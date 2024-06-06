#pragma once

#include "VulkanHandler.h"

#include <memory>
#include <unordered_map>
#include <vector>

class DescriptorSetLayout
{
public:
    class Builder
    {
    public:
        explicit Builder(VulkanHandler& device) : device{device} {}

        Builder& addBinding(
            uint32_t binding,
            VkDescriptorType descriptor_type,
            VkShaderStageFlags stage_flags,
            uint32_t count = 1);
        [[nodiscard]] std::unique_ptr<DescriptorSetLayout> build() const;

    private:
        VulkanHandler& device;
        std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings{};
    };

    DescriptorSetLayout(
            VulkanHandler& device, std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings);
    ~DescriptorSetLayout();
    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout &operator=(const DescriptorSetLayout&) = delete;

    VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptor_set_layout; }

private:
    VulkanHandler& device;
    VkDescriptorSetLayout descriptor_set_layout;
    std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings;

    friend class DescriptorWriter;
};

class DescriptorPool
{
public:
    class Builder
    {
    public:
        Builder(VulkanHandler& device) : device{device} {}

        Builder& addPoolSize(VkDescriptorType descriptor_type, uint32_t count);
        Builder& setPoolFlags(VkDescriptorPoolCreateFlags flags);
        Builder& setMaxSets(uint32_t count);
        std::unique_ptr<DescriptorPool> build() const;

    private:
        VulkanHandler& device;
        std::vector<VkDescriptorPoolSize> pool_sizes{};
        uint32_t max_sets = 1000;
        VkDescriptorPoolCreateFlags pool_flags = 0;
    };

    DescriptorPool(
            VulkanHandler& device,
            uint32_t max_sets,
            VkDescriptorPoolCreateFlags pool_flags,
            const std::vector<VkDescriptorPoolSize>& pool_sizes);
    ~DescriptorPool();

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool &operator=(const DescriptorPool&) = delete;

    VkDescriptorPool descriptorPool() { return descriptor_pool; }

    bool allocateDescriptor(const VkDescriptorSetLayout descriptor_set_layout, VkDescriptorSet& descriptor) const;
    void freeDescriptors(std::vector<VkDescriptorSet>& descriptors) const;
    void resetPool();

private:
    VulkanHandler& device;
    VkDescriptorPool descriptor_pool;

    friend class DescriptorWriter;
};

class DescriptorWriter
{
public:
    DescriptorWriter(DescriptorSetLayout& set_layout, DescriptorPool& pool);

    DescriptorWriter& writeBuffer(uint32_t binding, VkDescriptorBufferInfo* buffer_info);
    DescriptorWriter& writeImage(uint32_t binding, VkDescriptorImageInfo* image_info, uint32_t count = 1);
    DescriptorWriter& writeAccelerationStructure(uint32_t binding, VkWriteDescriptorSetAccelerationStructureKHR* structure_info);

    bool build(VkDescriptorSet& set);
    void overwrite(VkDescriptorSet& set);

private:
    DescriptorSetLayout& set_layout;
    DescriptorPool& pool;
    std::vector<VkWriteDescriptorSet> writes;
};