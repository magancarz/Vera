#include "gtest/gtest.h"

#include <TestUtils.h>

#include "RenderEngine/RenderingAPI/Descriptors/DescriptorSetLayoutBuilder.h"

TEST(DescriptorSetLayoutBuilderTests, shouldBuildInvalidDescriptorSetLayoutWhenGivenBindingsAreEmpty)
{
    // given
    auto descriptor_set_layout_builder = DescriptorSetLayoutBuilder(TestsEnvironment::vulkanHandler());

    // when
    auto descriptor_set_layout = descriptor_set_layout_builder.build();

    // then
    EXPECT_EQ(descriptor_set_layout, nullptr);
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}

TEST(DescriptorSetLayoutBuilderTests, shouldBuildValidDescriptorSetLayout)
{
    // given
    auto descriptor_set_layout_builder = DescriptorSetLayoutBuilder(TestsEnvironment::vulkanHandler())
        .addBinding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
        .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
        .addBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
        .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
        .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
        .addBinding(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

    // when
    auto descriptor_set_layout = descriptor_set_layout_builder.build();

    // then
    EXPECT_NE(descriptor_set_layout->getDescriptorSetLayout(), VK_NULL_HANDLE);
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}