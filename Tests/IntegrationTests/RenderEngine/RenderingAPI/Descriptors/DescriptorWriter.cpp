#include "gtest/gtest.h"

#include "RenderEngine/RenderingAPI/Descriptors/DescriptorWriter.h"
#include "TestUtils.h"
#include "Environment.h"
#include "RenderEngine/RenderingAPI/Descriptors/DescriptorPoolBuilder.h"
#include "RenderEngine/RenderingAPI/Descriptors/DescriptorSetLayoutBuilder.h"

TEST(DescriptorPoolBuilderTests, shouldWriteReferencesToDescriptorSet)
{
    // given
    auto descriptor_pool = DescriptorPoolBuilder(TestsEnvironment::vulkanHandler())
        .setMaxSets(2)
        .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)
        .build();

    auto descriptor_set_layout = DescriptorSetLayoutBuilder(TestsEnvironment::vulkanHandler())
        .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
        .build();

    BufferInfo uniform_buffer_info{};
    uniform_buffer_info.instance_size = sizeof(uint8_t);
    uniform_buffer_info.instance_count = 22;
    uniform_buffer_info.usage_flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto uniform_buffer = TestsEnvironment::memoryAllocator().createBuffer(uniform_buffer_info);

    auto camera_buffer_descriptor_info = uniform_buffer->descriptorInfo();

    VkDescriptorSet descriptor_set_handle;

    // when
    DescriptorWriter(*descriptor_set_layout, *descriptor_pool)
        .writeBuffer(0, &camera_buffer_descriptor_info)
        .build(descriptor_set_handle);

    // then
    EXPECT_NE(descriptor_set_handle, VK_NULL_HANDLE);
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}