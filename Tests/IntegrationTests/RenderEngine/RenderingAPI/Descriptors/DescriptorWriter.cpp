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

    auto uniform_buffer = TestsEnvironment::memoryAllocator().createBuffer(
        sizeof(uint8_t), 22, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

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