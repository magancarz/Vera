#include <RenderEngine/Pipeline/RayTracingPipelineBuilder.h>
#include <RenderEngine/RenderingAPI/Descriptors/DescriptorSetLayoutBuilder.h>

#include "gtest/gtest.h"

#include "TestUtils.h"
#include "Environment.h"

TEST(RayTracingPipelineBuilderTests, shouldBuildValidRayTracingPipeline)
{
    // given
    auto acceleration_structure_descriptor_set_layout = DescriptorSetLayoutBuilder(TestsEnvironment::vulkanHandler())
        .addBinding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
        .build();

    auto ray_traced_image_descriptor_set_layout = DescriptorSetLayoutBuilder(TestsEnvironment::vulkanHandler())
        .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
        .build();

    auto object_descriptions_descriptor_set_layout = DescriptorSetLayoutBuilder(TestsEnvironment::vulkanHandler())
        .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
        .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
        .addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
        .addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
        .addBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
        .build();

    auto ray_tracing_pipeline_builder = RayTracingPipelineBuilder(TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator())
        .addRayGenerationStage(std::make_unique<ShaderModule>(TestsEnvironment::vulkanHandler(), "raytrace", VK_SHADER_STAGE_RAYGEN_BIT_KHR))
        .addMissStage(std::make_unique<ShaderModule>(TestsEnvironment::vulkanHandler(), "raytrace", VK_SHADER_STAGE_MISS_BIT_KHR))
        .addMissStage(std::make_unique<ShaderModule>(TestsEnvironment::vulkanHandler(), "raytrace_shadow", VK_SHADER_STAGE_MISS_BIT_KHR))
        .addDefaultOcclusionCheckShader(std::make_unique<ShaderModule>(TestsEnvironment::vulkanHandler(), "raytrace_occlusion", VK_SHADER_STAGE_ANY_HIT_BIT_KHR))
        .addDescriptorSetLayout(acceleration_structure_descriptor_set_layout->getDescriptorSetLayout())
        .addDescriptorSetLayout(ray_traced_image_descriptor_set_layout->getDescriptorSetLayout())
        .addDescriptorSetLayout(object_descriptions_descriptor_set_layout->getDescriptorSetLayout())
        .setMaxRecursionDepth(2);

    ray_tracing_pipeline_builder.addMaterialShader(
        "lambertian",
        std::make_unique<ShaderModule>(TestsEnvironment::vulkanHandler(), "raytrace_lambertian", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
        std::make_unique<ShaderModule>(TestsEnvironment::vulkanHandler(), "raytrace_alpha_check", VK_SHADER_STAGE_ANY_HIT_BIT_KHR),
        std::make_unique<ShaderModule>(TestsEnvironment::vulkanHandler(), "raytrace_alpha_check", VK_SHADER_STAGE_ANY_HIT_BIT_KHR));
    ray_tracing_pipeline_builder.registerObjectMaterial("lambertian");

    // when
    auto ray_tracing_pipeline = ray_tracing_pipeline_builder.build();

    // then
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}