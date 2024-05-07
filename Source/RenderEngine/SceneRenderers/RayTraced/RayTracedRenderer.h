#pragma once

#include "RenderEngine/SceneRenderers/SceneRenderer.h"
#include "World/World.h"
#include "RenderEngine/SceneRenderers/RayTraced/Pipeline/RayTracingPipeline.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/Models/RayTracingAccelerationStructureBuilder.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"

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

    VkImageView getRayTracedImageViewHandle() { return ray_traced_texture->getImageView(); }
    VkSampler getRayTracedImageSampler() { return ray_traced_texture->getSampler(); }

private:
    Device& device;
    World* world;

    void queryRayTracingPipelineProperties();

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    void createAccelerationStructure();

    AccelerationStructure tlas{};

    void createRayTracedImage();

    std::unique_ptr<Texture> ray_traced_texture;

    void createCameraUniformBuffer();

    std::unique_ptr<Buffer> camera_uniform_buffer;

    void createObjectDescriptionsBuffer();

    std::unique_ptr<Buffer> object_descriptions_buffer;

    void createLightIndicesBuffer();

    std::unique_ptr<Buffer> light_indices_buffer;
    uint32_t number_of_lights{0};

    void createDescriptors();

    std::unique_ptr<DescriptorPool> descriptor_pool;
    std::unique_ptr<DescriptorSetLayout> descriptor_set_layout;
    VkDescriptorSet descriptor_set_handle;

    void createRayTracingPipeline();

    std::unique_ptr<RayTracingPipeline> ray_tracing_pipeline;

    void updatePipelineUniformVariables(FrameInfo& frame_info);
    void updateCameraUniformBuffer(FrameInfo& frame_info);
    void updateRayPushConstant(FrameInfo& frame_info);
    void executeRayTracing(FrameInfo& frame_info);

    uint32_t current_number_of_frames{0};

    ShaderBindingTableValues shader_binding_table;
};
