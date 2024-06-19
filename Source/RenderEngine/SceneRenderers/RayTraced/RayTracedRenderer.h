#pragma once

#include "RenderEngine/AccelerationStructures/ObjectDescription.h"
#include "RenderEngine/SceneRenderers/SceneRenderer.h"
#include "World/World.h"
#include "RenderEngine/Pipeline/RayTracingPipeline.h"
#include "RenderEngine/RenderingAPI/Descriptors/DescriptorSetLayout.h"
#include "RenderEngine/Textures/DeviceTexture.h"
#include "RenderEngine/AccelerationStructures/Blas.h"
#include "RenderEngine/AccelerationStructures/Tlas.h"
#include "RenderEngine/Materials/DeviceMaterialInfo.h"
#include "RenderEngine/Pipeline/Compute/ComputePipeline.h"

class DescriptorPool;
class DescriptorWriter;

class RayTracedRenderer : public SceneRenderer
{
public:
    RayTracedRenderer(VulkanHandler& device, MemoryAllocator& memory_allocator, AssetManager& asset_manager, World& world);

    void renderScene(FrameInfo& frame_info) override;

    void handleWindowResize(uint32_t new_width, uint32_t new_height);

    [[nodiscard]] const DeviceTexture& getRayTracedImage() const { return *ray_traced_texture; }

private:
    VulkanHandler& device;
    MemoryAllocator& memory_allocator;
    AssetManager& asset_manager;
    World& world;

    std::vector<Object*> obtainRenderedObjectsFromWorld();

    std::vector<Object*> rendered_objects;

    void createObjectAndMaterialDescriptions();
    std::vector<ObjectDescription> createObjectDescriptionsFrom(Object* rendered_object);

    std::vector<uint32_t> object_description_offsets;
    std::vector<ObjectDescription> object_descriptions;
    std::unique_ptr<Buffer> object_descriptions_buffer;

    uint32_t fetchRequiredMaterialIndex(const MeshComponent* mesh_component, const std::string_view& required_material_name);
    uint32_t fetchRequiredDiffuseTextureOffset(const Material* material);
    uint32_t fetchRequiredNormalMapOffset(const Material* material);

    std::vector<DeviceTexture*> diffuse_textures;
    std::vector<DeviceTexture*> normal_textures;
    std::vector<DeviceMaterialInfo> material_infos;
    std::unique_ptr<Buffer> material_descriptions_buffer;

    void createAccelerationStructure();
    std::vector<BlasInstance> getBlasInstances();

    std::unordered_map<std::string, Blas> blas_objects;
    std::unique_ptr<Tlas> tlas;

    void createRayTracedImageWithCurrentWindowSize();
    void createRayTracedImage(uint32_t width, uint32_t height);

    std::unique_ptr<DeviceTexture> ray_traced_texture;

    void createCameraUniformBuffer();

    std::unique_ptr<Buffer> camera_uniform_buffer;

    void createDescriptors();

    std::unique_ptr<DescriptorPool> descriptor_pool;
    std::unique_ptr<DescriptorWriter> descriptor_writer;

    void createAccelerationStructureDescriptor();
    void createAccelerationStructureDescriptorLayout();
    void writeToAccelerationStructureDescriptorSet();

    std::unique_ptr<DescriptorSetLayout> acceleration_structure_descriptor_set_layout;
    VkDescriptorSet acceleration_structure_descriptor_set_handle{VK_NULL_HANDLE};

    void createRayTracedImageDescriptor();
    void createRayTracedImageDescriptorLayout();
    void writeToRayTracedImageDescriptorSet();

    std::unique_ptr<DescriptorSetLayout> ray_traced_image_descriptor_set_layout;
    VkDescriptorSet ray_traced_image_descriptor_set_handle{VK_NULL_HANDLE};

    void createObjectDescriptionsDescriptor();
    void createObjectDescriptionsDescriptorLayout();
    void writeToObjectDescriptionsDescriptorSet();

    std::unique_ptr<DescriptorSetLayout> objects_descriptions_descriptor_set_layout;
    VkDescriptorSet objects_info_descriptor_set_handle{VK_NULL_HANDLE};

    void buildRayTracingPipeline();

    std::vector<Material*> used_materials;
    std::unique_ptr<RayTracingPipeline> ray_tracing_pipeline;

    void updatePipelineUniformVariables(FrameInfo& frame_info);
    void updateCameraUniformBuffer(FrameInfo& frame_info);
    void bindDescriptorSets(FrameInfo& frame_info);
    void updateRayPushConstant(FrameInfo& frame_info);
    void executeRayTracing(FrameInfo& frame_info);

    uint32_t current_number_of_frames{0};
};
