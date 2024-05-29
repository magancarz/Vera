#pragma once

#include "RenderEngine/SceneRenderers/SceneRenderer.h"
#include "World/World.h"
#include "RenderEngine/SceneRenderers/RayTraced/Pipeline/RayTracingPipeline.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/Models/RayTracingAccelerationStructureBuilder.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"
#include "RenderEngine/RenderingAPI/Blas.h"
#include "RenderEngine/SceneRenderers/RayTraced/Pipeline/RayTracingPipelineBuilder.h"

struct CameraUBO
{
    glm::mat4 camera_view{};
    glm::mat4 camera_proj{};
};

class RayTracedRenderer : public SceneRenderer
{
public:
    RayTracedRenderer(VulkanFacade& device, std::unique_ptr<MemoryAllocator>& memory_allocator, std::shared_ptr<AssetManager> asset_manager, World* world);
    ~RayTracedRenderer() noexcept override;

    void renderScene(FrameInfo& frame_info) override;

    VkImageView getRayTracedImageViewHandle() { return ray_traced_texture->getImageView(); }
    VkSampler getRayTracedImageSampler() { return ray_traced_texture->getSampler(); }

private:
    VulkanFacade& device;
    std::unique_ptr<MemoryAllocator>& memory_allocator;
    std::shared_ptr<AssetManager> asset_manager;
    World* world;

    void queryRayTracingPipelineProperties();

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    void createObjectDescriptionsBuffer();

    std::vector<std::shared_ptr<Texture>> diffuse_textures;
    std::vector<std::shared_ptr<Texture>> normal_textures;
    std::unique_ptr<Buffer> object_descriptions_buffer;
    std::unique_ptr<Buffer> material_descriptions_buffer;
    std::vector<uint32_t> object_description_offsets;

    void createAccelerationStructure();

    std::unordered_map<std::string, Blas> blas_objects;
    std::unordered_map<int, std::shared_ptr<Object>> rendered_objects;
    AccelerationStructure tlas{};

    void createRayTracedImage();

    std::unique_ptr<Texture> ray_traced_texture;

    void createCameraUniformBuffer();

    std::unique_ptr<Buffer> camera_uniform_buffer;

    void createDescriptors();

    std::unique_ptr<DescriptorPool> descriptor_pool;
    std::unique_ptr<DescriptorSetLayout> descriptor_set_layout;
    VkDescriptorSet descriptor_set_handle;

    void buildRayTracingPipeline();

    std::vector<ObjectDescription> object_descriptions;
    std::vector<std::shared_ptr<Material>> used_materials;
    std::unique_ptr<RayTracingPipeline> ray_tracing_pipeline;

    void updatePipelineUniformVariables(FrameInfo& frame_info);
    void updateCameraUniformBuffer(FrameInfo& frame_info);
    void updateRayPushConstant(FrameInfo& frame_info);
    void executeRayTracing(FrameInfo& frame_info);

    uint32_t current_number_of_frames{0};
};
