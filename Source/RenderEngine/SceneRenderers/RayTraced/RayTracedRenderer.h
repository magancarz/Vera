#pragma once

#include "RenderEngine/SceneRenderers/SceneRenderer.h"
#include "World/World.h"
#include "RayTracingPipeline.h"

struct CameraUBO
{
    float cameraPosition[4] = {0, 0, 0, 1};
    float cameraRight[4] = {1, 0, 0, 1};
    float cameraUp[4] = {0, 1, 0, 1};
    float cameraForward[4] = {0, 0, 1, 1};

    unsigned int frameCount = 0;
};

class RayTracedRenderer : public SceneRenderer
{
public:
    RayTracedRenderer(Device& device, World* world);

    void renderScene(FrameInfo& frame_info) override;

private:
    Device& device;
    World* world;

    RayTracingBuilder ray_tracing_builder;

    void queryRayTracingPipelineProperties();

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    void createAccelerationStructure();

    void createRayTracedImage();

    VkImage rayTraceImageHandle{VK_NULL_HANDLE};
    VkImageView rayTraceImageViewHandle{VK_NULL_HANDLE};

    void createCameraUniformBuffer();

    std::unique_ptr<Buffer> camera_uniform_buffer;

    void createDescriptors();

    std::unique_ptr<DescriptorPool> descriptor_pool;
    std::unique_ptr<DescriptorSetLayout> descriptor_set_layout;
    VkDescriptorSet descriptor_set_handle;

    std::unique_ptr<DescriptorSetLayout> material_descriptor_set_layout;
    VkDescriptorSet material_descriptor_set_handle;

    void createRayTracingPipeline();

    std::unique_ptr<RayTracingPipeline> ray_tracing_pipeline;
};
