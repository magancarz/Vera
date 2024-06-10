
#include "gtest/gtest.h"

#include "TestUtils.h"
#include "RenderEngine/RenderingAPI/ShaderModule.h"

TEST(ShaderModuleTests, shouldCreateValidShaderModuleFromShaderCode)
{
    // given
    const std::string real_world_shader_code_file = "raytrace_lambertian";

    // when
    auto shader_module = std::make_unique<ShaderModule>(TestsEnvironment::vulkanHandler(), real_world_shader_code_file, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

    // then
    EXPECT_NE(shader_module->getShaderModule(), VK_NULL_HANDLE);
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}