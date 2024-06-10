#pragma once

#include "RenderEngine/RenderingAPI/VulkanHandler.h"

class DescriptorPool;
class DescriptorSetLayout;

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
