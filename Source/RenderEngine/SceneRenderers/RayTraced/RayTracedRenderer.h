#pragma once

#include "RenderEngine/SceneRenderers/SceneRenderer.h"
#include "World/World.h"
#include "RayTracingPipeline.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/Models/RayTracingAccelerationStructureBuilder.h"

struct CameraUBO
{
    glm::mat4 camera_view{};
    glm::mat4 camera_proj{};
};

class RayTracedRenderer : public SceneRenderer
{
public:
    RayTracedRenderer(Device& device, World* world);

    void renderScene(FrameInfo& frame_info) override;

private:
    Device& device;
    World* world;

    void queryRayTracingPipelineProperties();

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    void createAccelerationStructure();

    AccelerationStructure tlas{};

    void createRayTracedImage();

    VkImage rayTraceImageHandle{VK_NULL_HANDLE};
    VkImageView rayTraceImageViewHandle{VK_NULL_HANDLE};

    void createCameraUniformBuffer();

    std::unique_ptr<Buffer> camera_uniform_buffer;

    void createObjectDescriptionsBuffer();

    std::unique_ptr<Buffer> object_descriptions_buffer;

    void createLightIndicesBuffer();

    std::unique_ptr<Buffer> light_indices_buffer;

    void createDescriptors();

    std::unique_ptr<DescriptorPool> descriptor_pool;
    std::unique_ptr<DescriptorSetLayout> descriptor_set_layout;
    VkDescriptorSet descriptor_set_handle;

    void createRayTracingPipeline();

    std::unique_ptr<RayTracingPipeline> ray_tracing_pipeline;

    uint32_t current_number_of_frames{0};
};
