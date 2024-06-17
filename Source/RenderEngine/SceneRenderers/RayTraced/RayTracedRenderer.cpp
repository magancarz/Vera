#include "RayTracedRenderer.h"

#include <chrono>
#include <iostream>

#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "RenderEngine/GlobalUBO.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/Pipeline/RayTracingPipelineBuilder.h"
#include "Objects/Components/MeshComponent.h"
#include "RenderEngine/Materials/DeviceMaterialInfo.h"
#include "Objects/Object.h"
#include "Objects/Components/TransformComponent.h"
#include "RenderEngine/AccelerationStructures/Tlas.h"
#include "RenderEngine/RenderingAPI/Descriptors/DescriptorPoolBuilder.h"
#include "RenderEngine/RenderingAPI/Descriptors/DescriptorSetLayoutBuilder.h"
#include "RenderEngine/RenderingAPI/Descriptors/DescriptorWriter.h"

RayTracedRenderer::RayTracedRenderer(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        AssetManager& asset_manager,
        World& world)
    : device{device}, memory_allocator{memory_allocator}, asset_manager{asset_manager}, world{world},
    tlas{device.getLogicalDevice(), device.getCommandPool(), memory_allocator, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR}
{
    obtainRenderedObjectsFromWorld();
    createObjectDescriptionsBuffer();
    createAccelerationStructure();

    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.getPhysicalDeviceHandle(), device.getSurfaceKHRHandle(), &surface_capabilities);
    createRayTracedImage(surface_capabilities.currentExtent.width, surface_capabilities.currentExtent.height);
    createCameraUniformBuffer();
    createDescriptors();
    buildRayTracingPipeline();
}

void RayTracedRenderer::obtainRenderedObjectsFromWorld()
{
    rendered_objects.clear();
    for (const auto& [id, object] : world.getObjects())
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
            object_description.vertex_address = mesh_description.model_descriptions[i].vertex_buffer->
                getBufferDeviceAddress();
            object_description.index_address = mesh_description.model_descriptions[i].index_buffer->
                getBufferDeviceAddress();
            object_description.object_to_world = object->getTransform();

            auto required_material = mesh_description.required_materials[i];
            uint32_t material_index;
            auto it = std::ranges::find_if(used_materials.begin(), used_materials.end(),
               [&](const Material* item)
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
        static_cast<uint32_t>(object_descriptions.size()),
        object_descriptions.data());

    BufferInfo object_descriptions_buffer_info{};
    object_descriptions_buffer_info.instance_size = sizeof(ObjectDescription);
    object_descriptions_buffer_info.instance_count = static_cast<uint32_t>(object_descriptions.size());
    object_descriptions_buffer_info.usage_flags =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    object_descriptions_buffer_info.required_memory_flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    object_descriptions_buffer = memory_allocator.createBuffer(object_descriptions_buffer_info);
    object_descriptions_buffer->copyFrom(*object_descriptions_staging_buffer);

    auto material_descriptions_staging_buffer = memory_allocator.createStagingBuffer(
        sizeof(DeviceMaterialInfo),
        static_cast<uint32_t>(device_material_infos.size()),
        device_material_infos.data());

    BufferInfo material_descriptions_buffer_info{};
    material_descriptions_buffer_info.instance_size = sizeof(DeviceMaterialInfo);
    material_descriptions_buffer_info.instance_count = static_cast<uint32_t>(device_material_infos.size());
    material_descriptions_buffer_info.usage_flags =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    material_descriptions_buffer_info.required_memory_flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    material_descriptions_buffer = memory_allocator.createBuffer(material_descriptions_buffer_info);
    material_descriptions_buffer->copyFrom(*material_descriptions_staging_buffer);
}

void RayTracedRenderer::createAccelerationStructure()
{
    std::vector<BlasInstance> blas_instances = getBlasInstances();
    tlas.build(blas_instances);
}

std::vector<BlasInstance> RayTracedRenderer::getBlasInstances()
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
                std::forward_as_tuple(device, memory_allocator, asset_manager, *mesh_component->getMesh()));
        }
        BlasInstance blas_instance = blas_objects.at(mesh_component->getMeshName()).createBlasInstance(object->getTransform());
        blas_instance.bottom_level_acceleration_structure_instance.instanceCustomIndex = object_description_offsets[i++];
        blas_instances.push_back(std::move(blas_instance));
    }

    return blas_instances;
}

void RayTracedRenderer::createRayTracedImage(uint32_t width, uint32_t height)
{
    TextureData texture_data{};
    texture_data.name = "ray_traced_texture";
    texture_data.width = width;
    texture_data.height = height;
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
    BufferInfo material_descriptions_buffer_info{};
    material_descriptions_buffer_info.instance_size = sizeof(GlobalUBO);
    material_descriptions_buffer_info.instance_count = 1;
    material_descriptions_buffer_info.usage_flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    material_descriptions_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    material_descriptions_buffer_info.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    camera_uniform_buffer = memory_allocator.createBuffer(material_descriptions_buffer_info);
    camera_uniform_buffer->map();
}

void RayTracedRenderer::createDescriptors()
{
    descriptor_pool = DescriptorPoolBuilder(device)
        .setMaxSets(4)
        .addPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1)
        .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)
        .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5)
        .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1)
        .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(diffuse_textures.size() + normal_textures.size()))
        .build();

    createAccelerationStructureDescriptor();
    createRayTracedImageDescriptor();
    createObjectDescriptionsDescriptor();
}

void RayTracedRenderer::createAccelerationStructureDescriptor()
{
    createAccelerationStructureDescriptorLayout();
    writeToAccelerationStructureDescriptorSet();
}

void RayTracedRenderer::createAccelerationStructureDescriptorLayout()
{
    acceleration_structure_descriptor_set_layout = DescriptorSetLayoutBuilder(device)
        .addBinding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
        .build();
}

void RayTracedRenderer::writeToAccelerationStructureDescriptorSet()
{
    VkWriteDescriptorSetAccelerationStructureKHR acceleration_structure_descriptor_info{};
    acceleration_structure_descriptor_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    acceleration_structure_descriptor_info.accelerationStructureCount = 1;
    acceleration_structure_descriptor_info.pAccelerationStructures = &tlas.accelerationStructure().handle;

    auto descriptor_writer = DescriptorWriter(*acceleration_structure_descriptor_set_layout, *descriptor_pool)
        .writeAccelerationStructure(0, &acceleration_structure_descriptor_info);

    if (acceleration_structure_descriptor_set_handle == VK_NULL_HANDLE)
    {
        descriptor_writer.build(acceleration_structure_descriptor_set_handle);
    }
    else
    {
        descriptor_writer.overwrite(acceleration_structure_descriptor_set_handle);
    }
}

void RayTracedRenderer::createRayTracedImageDescriptor()
{
    createRayTracedImageDescriptorLayout();
    writeToRayTracedImageDescriptorSet();
}

void RayTracedRenderer::createRayTracedImageDescriptorLayout()
{
    ray_traced_image_descriptor_set_layout = DescriptorSetLayoutBuilder(device)
        .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
        .build();
}

void RayTracedRenderer::writeToRayTracedImageDescriptorSet()
{
    VkDescriptorImageInfo ray_traced_image_descriptor_info = ray_traced_texture->descriptorInfo();

    auto descriptor_writer = DescriptorWriter(*ray_traced_image_descriptor_set_layout, *descriptor_pool)
        .writeImage(0, &ray_traced_image_descriptor_info);

    if (ray_traced_image_descriptor_set_handle == VK_NULL_HANDLE)
    {
        descriptor_writer.build(ray_traced_image_descriptor_set_handle);
    }
    else
    {
        descriptor_writer.overwrite(ray_traced_image_descriptor_set_handle);
    }
}

void RayTracedRenderer::createObjectDescriptionsDescriptor()
{
    createObjectDescriptionsDescriptorLayout();
    writeToObjectDescriptionsDescriptorSet();
}

void RayTracedRenderer::createObjectDescriptionsDescriptorLayout()
{
    objects_descriptions_descriptor_set_layout = DescriptorSetLayoutBuilder(device)
        .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
        .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
        .addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
        .addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
            static_cast<uint32_t>(diffuse_textures.size()))
        .addBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
            static_cast<uint32_t>(normal_textures.size()))
        .build();
}

void RayTracedRenderer::writeToObjectDescriptionsDescriptorSet()
{
    auto camera_buffer_descriptor_info = camera_uniform_buffer->descriptorInfo();
    auto object_descriptions_buffer_descriptor_info = object_descriptions_buffer->descriptorInfo();
    auto material_descriptions_descriptor_info = material_descriptions_buffer->descriptorInfo();

    std::vector<VkDescriptorImageInfo> diffuse_texture_descriptor_infos;
    diffuse_texture_descriptor_infos.reserve(diffuse_textures.size());
    for (auto& texture : diffuse_textures)
    {
        diffuse_texture_descriptor_infos.emplace_back(texture->descriptorInfo());
    }

    std::vector<VkDescriptorImageInfo> normal_texture_descriptor_infos;
    normal_texture_descriptor_infos.reserve(normal_textures.size());
    for (auto& texture : normal_textures)
    {
        normal_texture_descriptor_infos.emplace_back(texture->descriptorInfo());
    }

    auto descriptor_writer = DescriptorWriter(*objects_descriptions_descriptor_set_layout, *descriptor_pool)
        .writeBuffer(0, &camera_buffer_descriptor_info)
        .writeBuffer(1, &object_descriptions_buffer_descriptor_info)
        .writeBuffer(2, &material_descriptions_descriptor_info)
        .writeImage(3, diffuse_texture_descriptor_infos.data(), static_cast<uint32_t>(diffuse_texture_descriptor_infos.size()))
        .writeImage(4, normal_texture_descriptor_infos.data(), static_cast<uint32_t>(normal_texture_descriptor_infos.size()));

    if (objects_info_descriptor_set_handle == VK_NULL_HANDLE)
    {
        descriptor_writer.build(objects_info_descriptor_set_handle);
    }
    else
    {
        descriptor_writer.overwrite(objects_info_descriptor_set_handle);
    }
}

void RayTracedRenderer::buildRayTracingPipeline()
{
    auto ray_tracing_pipeline_builder = RayTracingPipelineBuilder(device, memory_allocator)
        .addRayGenerationStage(std::make_unique<ShaderModule>(device, "raytrace", VK_SHADER_STAGE_RAYGEN_BIT_KHR))
        .addMissStage(std::make_unique<ShaderModule>(device, "raytrace", VK_SHADER_STAGE_MISS_BIT_KHR))
        .addMissStage(std::make_unique<ShaderModule>(device, "raytrace_shadow", VK_SHADER_STAGE_MISS_BIT_KHR))
        .addDefaultOcclusionCheckShader(std::make_unique<ShaderModule>(device, "raytrace_occlusion", VK_SHADER_STAGE_ANY_HIT_BIT_KHR))
        .addDescriptorSetLayout(acceleration_structure_descriptor_set_layout->getDescriptorSetLayout())
        .addDescriptorSetLayout(ray_traced_image_descriptor_set_layout->getDescriptorSetLayout())
        .addDescriptorSetLayout(objects_descriptions_descriptor_set_layout->getDescriptorSetLayout())
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
    pvkDestroyAccelerationStructureKHR(device.getDeviceHandle(), tlas.accelerationStructure().handle, VulkanDefines::NO_CALLBACK);
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
    bindDescriptorSets(frame_info);
    updateRayPushConstant(frame_info);
}

void RayTracedRenderer::updateCameraUniformBuffer(FrameInfo& frame_info)
{
    GlobalUBO camera_ubo{};
    camera_ubo.view = frame_info.camera_view_matrix;
    camera_ubo.projection = frame_info.camera_projection_matrix;
    camera_uniform_buffer->writeToBuffer(&camera_ubo);
}

void RayTracedRenderer::bindDescriptorSets(FrameInfo& frame_info)
{
    std::vector<VkDescriptorSet> descriptor_sets
    {
        acceleration_structure_descriptor_set_handle,
        ray_traced_image_descriptor_set_handle,
        objects_info_descriptor_set_handle
    };
    ray_tracing_pipeline->bindDescriptorSets(frame_info.command_buffer, descriptor_sets);
}

void RayTracedRenderer::updateRayPushConstant(FrameInfo& frame_info)
{
    PushConstantRay push_constant_ray{};
    push_constant_ray.time = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
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

void RayTracedRenderer::handleWindowResize(uint32_t new_width, uint32_t new_height)
{
    createRayTracedImage(new_width, new_height);
    writeToRayTracedImageDescriptorSet();
    current_number_of_frames = 0;
}
