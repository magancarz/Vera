#pragma once

#include "RenderEngine/AccelerationStructures/ObjectDescription.h"
#include "RenderEngine/SceneRenderers/SceneRenderer.h"
#include "World/World.h"
#include "RenderEngine/Pipeline/RayTracingPipeline.h"
#include "RenderEngine/RenderingAPI/Descriptors/DescriptorPool.h"
#include "RenderEngine/RenderingAPI/Descriptors/DescriptorSetLayout.h"
#include "RenderEngine/Textures/DeviceTexture.h"
#include "RenderEngine/AccelerationStructures/Blas.h"
#include "RenderEngine/Pipeline/RayTracingPipelineBuilder.h"

class RayTracedRenderer : public SceneRenderer
{
public:
    RayTracedRenderer(VulkanHandler& device, MemoryAllocator& memory_allocator, AssetManager& asset_manager, World& world);
    ~RayTracedRenderer() noexcept override;

    void renderScene(FrameInfo& frame_info) override;

    void recreateRayTracedImage(uint32_t new_width, uint32_t new_height);

    [[nodiscard]] const DeviceTexture& getRayTracedImage() const { return *ray_traced_texture; }

private:
    VulkanHandler& device;
    MemoryAllocator& memory_allocator;
    AssetManager& asset_manager;
    World& world;

    void queryRayTracingPipelineProperties();

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    void obtainRenderedObjectsFromWorld();

    std::vector<Object*> rendered_objects;

    void createObjectDescriptionsBuffer();

    std::vector<DeviceTexture*> diffuse_textures;
    std::vector<DeviceTexture*> normal_textures;
    std::unique_ptr<Buffer> object_descriptions_buffer;
    std::unique_ptr<Buffer> material_descriptions_buffer;
    std::vector<uint32_t> object_description_offsets;

    void createAccelerationStructure();

    std::unordered_map<std::string, Blas> blas_objects;
    AccelerationStructure tlas{};

    void createRayTracedImage(uint32_t width, uint32_t height);

    std::unique_ptr<DeviceTexture> ray_traced_texture;

    void createCameraUniformBuffer();

    std::unique_ptr<Buffer> camera_uniform_buffer;

    void createDescriptors();

    std::unique_ptr<DescriptorPool> descriptor_pool;
    std::unique_ptr<DescriptorSetLayout> descriptor_set_layout;
    VkDescriptorSet descriptor_set_handle;

    void buildRayTracingPipeline();

    std::vector<ObjectDescription> object_descriptions;
    std::vector<Material*> used_materials;
    std::unique_ptr<RayTracingPipeline> ray_tracing_pipeline;

    void updatePipelineUniformVariables(FrameInfo& frame_info);
    void updateCameraUniformBuffer(FrameInfo& frame_info);
    void updateRayPushConstant(FrameInfo& frame_info);
    void executeRayTracing(FrameInfo& frame_info);

    uint32_t current_number_of_frames{0};
};
