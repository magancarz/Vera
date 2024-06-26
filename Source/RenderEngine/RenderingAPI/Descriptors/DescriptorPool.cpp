#include "DescriptorPool.h"

DescriptorPool::DescriptorPool(
        VulkanHandler& device,
        uint32_t max_sets,
        VkDescriptorPoolCreateFlags pool_flags,
        const std::vector<VkDescriptorPoolSize>& pool_sizes)
        : device{device}
{
    VkDescriptorPoolCreateInfo descriptor_pool_info{};
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    descriptor_pool_info.pPoolSizes = pool_sizes.data();
    descriptor_pool_info.maxSets = max_sets;
    descriptor_pool_info.flags = pool_flags;

    if (vkCreateDescriptorPool(device.getDeviceHandle(), &descriptor_pool_info, nullptr, &descriptor_pool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

DescriptorPool::~DescriptorPool()
{
    vkDestroyDescriptorPool(device.getDeviceHandle(), descriptor_pool, nullptr);
}

bool DescriptorPool::allocateDescriptor(VkDescriptorSetLayout descriptor_set_layout, VkDescriptorSet& descriptor) const
{
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.pSetLayouts = &descriptor_set_layout;
    alloc_info.descriptorSetCount = 1;

    if (vkAllocateDescriptorSets(device.getDeviceHandle(), &alloc_info, &descriptor) != VK_SUCCESS)
    {
        return false;
    }
    return true;
}

void DescriptorPool::freeDescriptors(std::vector<VkDescriptorSet>& descriptors) const
{
    vkFreeDescriptorSets(
        device.getDeviceHandle(),
        descriptor_pool,
        static_cast<uint32_t>(descriptors.size()),
        descriptors.data());
}

void DescriptorPool::resetPool()
{
    vkResetDescriptorPool(device.getDeviceHandle(), descriptor_pool, 0);
}