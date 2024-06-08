#include "RayTracedRenderer.h"

#include <chrono>
#include <iostream>
#include <RenderEngine/AccelerationStructures/TlasBuilder.h>

#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "RenderEngine/GlobalUBO.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/SceneRenderers/RayTraced/Pipeline/RayTracingPipelineBuilder.h"
#include "Objects/Components/MeshComponent.h"
#include "RenderEngine/Materials/DeviceMaterialInfo.h"
#include "Objects/Object.h"

RayTracedRenderer::RayTracedRenderer(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        AssetManager& asset_manager,
        World& world)
    : device{device}, memory_allocator{memory_allocator}, asset_manager{asset_manager}, world{world}
{
    queryRayTracingPipelineProperties();
    obtainRenderedObjectsFromWorld();
    createObjectDescriptionsBuffer();
    createAccelerationStructure();
    createRayTracedImage();
    createCameraUniformBuffer();
    createDescriptors();
    buildRayTracingPipeline();
}

void RayTracedRenderer::queryRayTracingPipelineProperties()
{
    VkPhysicalDevice physical_device = device.getPhysicalDeviceHandle();

    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(physical_device,
                                  &physical_device_properties);

    ray_tracing_properties = VkPhysicalDeviceRayTracingPipelinePropertiesKHR{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 physical_device_properties_2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    physical_device_properties_2.pNext = &ray_tracing_properties;
    physical_device_properties_2.properties = physical_device_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &physical_device_properties_2);
}

void RayTracedRenderer::obtainRenderedObjectsFromWorld()
{
    rendered_objects.clear();
    for (auto& [id, object] : world.getObjects())
    {
        if (object->findComponentByClass<MeshComponent>())
        {
            rendered_objects.emplace_back(object.get());
        }
    }
}

void RayTracedRenderer::createObjectDescriptionsBuffer()
{
    std::vector<DeviceMaterialInfo> device_material_infos;
    diffuse_textures.clear();
    for (auto& object : rendered_objects)
    {
        object_description_offsets.emplace_back(static_cast<uint32_t>(object_descriptions.size()));
        auto mesh_component = object->findComponentByClass<MeshComponent>();
        MeshDescription mesh_description = mesh_component->getDescription();
        for (size_t i = 0; i < mesh_description.model_descriptions.size(); ++i)
        {
            ObjectDescription object_description{};
            object_description.vertex_address = mesh_description.model_descriptions[i].vertex_buffer->getBufferDeviceAddress();
            object_description.index_address = mesh_description.model_descriptions[i].index_buffer->getBufferDeviceAddress();
            object_description.object_to_world = object->getTransform();

            auto required_material = mesh_description.required_materials[i];
            uint32_t material_index;
            auto it = std::ranges::find_if(used_materials.begin(), used_materials.end(),
                   [&] (const Material* item)
                   {
                       return required_material == item->getName();
                   });
            if (it != used_materials.end())
            {
                material_index = std::distance(used_materials.begin(), it);
            }
            else
            {
                auto material = mesh_component->findMaterial(required_material);
                material_index = used_materials.size();
                used_materials.emplace_back(material);

                DeviceMaterialInfo device_material_info{};
                auto texture = material->getDiffuseTexture();
                auto tit = std::ranges::find(diffuse_textures.begin(), diffuse_textures.end(), texture);
                if (tit != diffuse_textures.end())
                {
                    device_material_info.diffuse_texture_offset = std::distance(diffuse_textures.begin(), tit);
                }
                else
                {
                    device_material_info.diffuse_texture_offset = diffuse_textures.size();
                    diffuse_textures.emplace_back(texture);
                }

                auto normal_texture = material->getNormalTexture();
                auto nit = std::ranges::find(normal_textures.begin(), normal_textures.end(), normal_texture);
                if (nit != normal_textures.end())
                {
                    device_material_info.normal_texture_offset = std::distance(normal_textures.begin(), nit);
                }
                else
                {
                    device_material_info.normal_texture_offset = normal_textures.size();
                    normal_textures.emplace_back(normal_texture);
                }

                device_material_infos.emplace_back(device_material_info);
            }

            object_description.material_index = material_index;
            object_descriptions.emplace_back(object_description);
        }
    }

    auto object_descriptions_staging_buffer = memory_allocator.createStagingBuffer(
            sizeof(ObjectDescription),
            object_descriptions.size(),
            object_descriptions.data());
    object_descriptions_buffer = memory_allocator.createBuffer
    (
            sizeof(ObjectDescription),
            static_cast<uint32_t>(object_descriptions.size()),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    object_descriptions_buffer->copyFrom(*object_descriptions_staging_buffer);

    auto material_descriptions_staging_buffer = memory_allocator.createStagingBuffer(
            sizeof(DeviceMaterialInfo),
            device_material_infos.size(),
            device_material_infos.data());
    material_descriptions_buffer = memory_allocator.createBuffer
    (
            sizeof(DeviceMaterialInfo),
            static_cast<uint32_t>(device_material_infos.size()),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    material_descriptions_buffer->copyFrom(*material_descriptions_staging_buffer);
}

void RayTracedRenderer::createAccelerationStructure()
{
    std::vector<BlasInstance> blas_instances;
    blas_instances.reserve(rendered_objects.size());
    size_t i = 0;
    for (auto& object : rendered_objects)
    {
        auto mesh_component = object->findComponentByClass<MeshComponent>();
        assert(mesh_component && "Rendered object must have mesh component!");

        if (!blas_objects.contains(mesh_component->getMeshName()))
        {
            blas_objects.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(mesh_component->getMeshName()),
                    std::forward_as_tuple(device, memory_allocator, asset_manager, *mesh_component));
        }
        BlasInstance blas_instance = blas_objects.at(mesh_component->getMeshName()).createBlasInstance(object->getTransform());
        blas_instance.bottomLevelAccelerationStructureInstance.instanceCustomIndex = object_description_offsets[i++];
        blas_instances.push_back(std::move(blas_instance));
    }
    tlas = TlasBuilder::buildTopLevelAccelerationStructure(device, memory_allocator, blas_instances);
}

void RayTracedRenderer::createRayTracedImage()
{
    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.getPhysicalDeviceHandle(), device.getSurfaceKHRHandle(), &surface_capabilities);

    TextureData texture_data{};
    texture_data.name = "ray_traced_texture";
    texture_data.width = surface_capabilities.currentExtent.width;
    texture_data.height = surface_capabilities.currentExtent.height;
    texture_data.mip_levels = 1;
    texture_data.number_of_channels = 4;
    texture_data.format = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkImageCreateInfo image_create_info{};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.extent.width = texture_data.width;
    image_create_info.extent.height = texture_data.height;
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = texture_data.mip_levels;
    image_create_info.arrayLayers = 1;
    image_create_info.format = texture_data.format;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    std::unique_ptr<Image> image = memory_allocator.createImage(image_create_info);

    ray_traced_texture = std::make_unique<DeviceTexture>(device, std::move(texture_data), std::move(image));
}

void RayTracedRenderer::createCameraUniformBuffer()
{
    camera_uniform_buffer = memory_allocator.createBuffer
    (
            sizeof(GlobalUBO),
            1,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    camera_uniform_buffer->map();
}

void RayTracedRenderer::createDescriptors()
{
    descriptor_pool = DescriptorPool::Builder(device)
            .setMaxSets(2)
            .addPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4)
            .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, diffuse_textures.size() + normal_textures.size())
            .build();

    descriptor_set_layout = DescriptorSetLayout::Builder(device)
            .addBinding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
            .addBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
            .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
            .addBinding(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR, diffuse_textures.size())
            .addBinding(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, normal_textures.size())
            .build();

    VkWriteDescriptorSetAccelerationStructureKHR acceleration_structure_descriptor_info{};
    acceleration_structure_descriptor_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    acceleration_structure_descriptor_info.accelerationStructureCount = 1;
    acceleration_structure_descriptor_info.pAccelerationStructures = &tlas.acceleration_structure;

    auto camera_buffer_descriptor_info = camera_uniform_buffer->descriptorInfo();
    auto object_descriptions_buffer_descriptor_info = object_descriptions_buffer->descriptorInfo();
    auto material_descriptions_descriptor_info = material_descriptions_buffer->descriptorInfo();

    VkDescriptorImageInfo ray_trace_image_descriptor_info = {
            .sampler = VK_NULL_HANDLE,
            .imageView = ray_traced_texture->getImageView(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    std::vector<VkDescriptorImageInfo> diffuse_texture_descriptor_infos;
    diffuse_texture_descriptor_infos.reserve(diffuse_textures.size());
    for (auto& texture : diffuse_textures)
    {
        VkDescriptorImageInfo texture_descriptor_info{};
        texture_descriptor_info.sampler = texture->getSampler();
        texture_descriptor_info.imageView = texture->getImageView();
        texture_descriptor_info.imageLayout = texture->getImageLayout();
        diffuse_texture_descriptor_infos.emplace_back(texture_descriptor_info);
    }

    std::vector<VkDescriptorImageInfo> normal_texture_descriptor_infos;
    normal_texture_descriptor_infos.reserve(normal_textures.size());
    for (auto& texture : normal_textures)
    {
        VkDescriptorImageInfo texture_descriptor_info{};
        texture_descriptor_info.sampler = texture->getSampler();
        texture_descriptor_info.imageView = texture->getImageView();
        texture_descriptor_info.imageLayout = texture->getImageLayout();
        normal_texture_descriptor_infos.emplace_back(texture_descriptor_info);
    }

    DescriptorWriter(*descriptor_set_layout, *descriptor_pool)
            .writeAccelerationStructure(0, &acceleration_structure_descriptor_info)
            .writeImage(1, &ray_trace_image_descriptor_info)
            .writeBuffer(2, &camera_buffer_descriptor_info)
            .writeBuffer(3, &object_descriptions_buffer_descriptor_info)
            .writeBuffer(4, &material_descriptions_descriptor_info)
            .writeImage(5, diffuse_texture_descriptor_infos.data(), diffuse_texture_descriptor_infos.size())
            .writeImage(6, normal_texture_descriptor_infos.data(), normal_texture_descriptor_infos.size())
            .build(descriptor_set_handle);
}

void RayTracedRenderer::buildRayTracingPipeline()
{
    auto ray_tracing_pipeline_builder = RayTracingPipelineBuilder(device, memory_allocator, ray_tracing_properties)
            .addRayGenerationStage(std::make_unique<ShaderModule>(device, "raytrace", VK_SHADER_STAGE_RAYGEN_BIT_KHR))
            .addMissStage(std::make_unique<ShaderModule>(device, "raytrace", VK_SHADER_STAGE_MISS_BIT_KHR))
            .addMissStage(std::make_unique<ShaderModule>(device, "raytrace_shadow", VK_SHADER_STAGE_MISS_BIT_KHR))
            .addDefaultOcclusionCheckShader(std::make_unique<ShaderModule>(device, "raytrace_occlusion", VK_SHADER_STAGE_ANY_HIT_BIT_KHR))
            .addDescriptorSetLayout(descriptor_set_layout->getDescriptorSetLayout())
            .setMaxRecursionDepth(2);

    ray_tracing_pipeline_builder.addMaterialShader(
            "lambertian",
            std::make_unique<ShaderModule>(device, "raytrace_lambertian", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
            std::make_unique<ShaderModule>(device, "raytrace_alpha_check", VK_SHADER_STAGE_ANY_HIT_BIT_KHR),
            std::make_unique<ShaderModule>(device, "raytrace_alpha_check", VK_SHADER_STAGE_ANY_HIT_BIT_KHR));
    for (size_t i = 0; i < object_descriptions.size(); ++i)
    {
        ray_tracing_pipeline_builder.registerObjectMaterial("lambertian");
    }

    ray_tracing_pipeline = ray_tracing_pipeline_builder.build();
}

RayTracedRenderer::~RayTracedRenderer() noexcept
{
    pvkDestroyAccelerationStructureKHR(device.getDeviceHandle(), tlas.acceleration_structure, VulkanDefines::NO_CALLBACK);
}

void RayTracedRenderer::renderScene(FrameInfo& frame_info)
{
    ray_tracing_pipeline->bind(frame_info.command_buffer);
    updatePipelineUniformVariables(frame_info);
    executeRayTracing(frame_info);
}

void RayTracedRenderer::updatePipelineUniformVariables(FrameInfo& frame_info)
{
    updateCameraUniformBuffer(frame_info);
    updateRayPushConstant(frame_info);
}

void RayTracedRenderer::updateCameraUniformBuffer(FrameInfo& frame_info)
{
    GlobalUBO camera_ubo{};
    camera_ubo.view = frame_info.camera_view_matrix;
    camera_ubo.projection = frame_info.camera_projection_matrix;
    camera_uniform_buffer->writeToBuffer(&camera_ubo);
    ray_tracing_pipeline->bindDescriptorSets(frame_info.command_buffer, {descriptor_set_handle});
}

void RayTracedRenderer::updateRayPushConstant(FrameInfo& frame_info)
{
    PushConstantRay push_constant_ray{};
    push_constant_ray.time = std::chrono::system_clock::now().time_since_epoch().count();
    current_number_of_frames = frame_info.need_to_refresh_generated_image ? 0 : current_number_of_frames;
    push_constant_ray.frames = current_number_of_frames;
    push_constant_ray.weather = frame_info.weather;
    push_constant_ray.sun_position = frame_info.sun_position;
    ray_tracing_pipeline->pushConstants(frame_info.command_buffer, push_constant_ray);
}

void RayTracedRenderer::executeRayTracing(FrameInfo& frame_info)
{
    ShaderBindingTableValues shader_binding_table = ray_tracing_pipeline->getShaderBindingTableValues();
    constexpr uint32_t DEPTH = 1;
    pvkCmdTraceRaysKHR(frame_info.command_buffer,
            &shader_binding_table.ray_gen_shader_binding_table,
            &shader_binding_table.miss_shader_binding_table,
            &shader_binding_table.closest_hit_shader_binding_table,
            &shader_binding_table.callable_shader_binding_table,
            frame_info.window_size.width, frame_info.window_size.height, DEPTH);

    ++current_number_of_frames;
}