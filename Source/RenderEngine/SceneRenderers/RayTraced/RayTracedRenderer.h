#pragma once

#include "RenderEngine/SceneRenderers/SceneRenderer.h"
#include "World/World.h"
#include "RayTracingPipeline.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"

struct CameraUBO
{
    glm::mat4 camera_view{};
    glm::mat4 camera_proj{};

    unsigned int frame_count = 0;
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

    //TODO: move to asset manager
    void createMaterialsBuffer();

    std::unique_ptr<Buffer> material_uniform_buffer;
    std::vector<Material> materials;

    void createDescriptors();

    std::unique_ptr<DescriptorPool> descriptor_pool;
    std::unique_ptr<DescriptorSetLayout> descriptor_set_layout;
    VkDescriptorSet descriptor_set_handle;

    std::unique_ptr<DescriptorSetLayout> material_descriptor_set_layout;
    VkDescriptorSet material_descriptor_set_handle;

    void createRayTracingPipeline();

    std::unique_ptr<RayTracingPipeline> ray_tracing_pipeline;
};
