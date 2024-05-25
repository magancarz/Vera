#include "RayTracedRenderer.h"

#include <chrono>
#include <iostream>

#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "RenderEngine/GlobalUBO.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/SceneRenderers/RayTraced/Pipeline/RayTracingPipelineBuilder.h"
#include "Objects/Components/MeshComponent.h"

RayTracedRenderer::RayTracedRenderer(VulkanFacade& device, World* world)
    : device{device}, world{world}
{
    queryRayTracingPipelineProperties();
    createObjectDescriptionsBuffer();
    createAccelerationStructure();
    createRayTracedImage();
    createCameraUniformBuffer();
    createLightIndicesBuffer();
    createDescriptors();
    createRayTracingPipeline();
}

void RayTracedRenderer::queryRayTracingPipelineProperties()
{
    VkPhysicalDevice physical_device = device.getPhysicalDevice();

    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(physical_device,
                                  &physical_device_properties);

    ray_tracing_properties = VkPhysicalDeviceRayTracingPipelinePropertiesKHR{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 physical_device_properties_2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    physical_device_properties_2.pNext = &ray_tracing_properties;
    physical_device_properties_2.properties = physical_device_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &physical_device_properties_2);
}

void RayTracedRenderer::createObjectDescriptionsBuffer()
{
    std::vector<ObjectDescription> object_descriptions;
    std::vector<std::shared_ptr<Material>> used_materials;
    std::vector<MaterialInfo> material_infos;
    textures.clear();
    rendered_objects = world->getRenderedObjects();
    for (auto& [_, object] : rendered_objects)
    {
        object_description_offsets.emplace_back(object_descriptions.size());
        auto mesh_component = object->findComponentByClass<MeshComponent>();
        MeshDescription mesh_description = mesh_component->getDescription();
        for (size_t i = 0; i < mesh_description.model_descriptions.size(); ++i)
        {
            ObjectDescription object_description{};
            object_description.vertex_address = mesh_description.model_descriptions[i].vertex_address;
            object_description.index_address = mesh_description.model_descriptions[i].index_address;
            object_description.object_to_world = object->getTransform();

            auto required_material = mesh_description.required_materials[i];
            uint32_t material_index{0};
            auto it = std::find_if(used_materials.begin(), used_materials.end(),
                   [&] (const std::shared_ptr<Material>& item)
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
                auto texture = material->getTexture();
                used_materials.emplace_back(material);
                auto material_info = material->getMaterialInfo();
                auto tit = std::find(textures.begin(), textures.end(), texture);
                if (tit != textures.end())
                {
                    material_info.texture_offset = std::distance(textures.begin(), tit);
                }
                else
                {
                    material_info.texture_offset = textures.size();
                    textures.emplace_back(texture);
                }
                material_infos.emplace_back(material_info);
            }

            object_description.material_index = material_index;
            object_descriptions.emplace_back(object_description);
        }
    }

    object_descriptions_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(ObjectDescription),
            static_cast<uint32_t>(object_descriptions.size()),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    object_descriptions_buffer->writeWithStagingBuffer(object_descriptions.data());

    material_descriptions_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(MaterialInfo),
            static_cast<uint32_t>(material_infos.size()),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    material_descriptions_buffer->writeWithStagingBuffer(material_infos.data());
}

void RayTracedRenderer::createAccelerationStructure()
{
    RayTracingAccelerationStructureBuilder builder{device};

    std::vector<BlasInstance> blas_instances;
    blas_instances.reserve(rendered_objects.size());
    size_t i = 0;
    for (auto [_, object] : rendered_objects)
    {
        auto mesh_component = object->findComponentByClass<MeshComponent>();

        if (!blas_objects.contains(mesh_component->getModel()->getName()))
        {
            blas_objects.emplace(std::piecewise_construct, std::forward_as_tuple(mesh_component->getModel()->getName()), std::forward_as_tuple(device, mesh_component));
        }
        BlasInstance blas_instance = blas_objects.at(mesh_component->getModel()->getName()).createBlasInstance(object->getTransform());
        blas_instance.bottomLevelAccelerationStructureInstance.instanceCustomIndex = object_description_offsets[i++];
        blas_instances.push_back(std::move(blas_instance));
    }
    tlas = builder.buildTopLevelAccelerationStructure(blas_instances);
}

void RayTracedRenderer::createRayTracedImage()
{
    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.getPhysicalDevice(), device.getSurface(), &surface_capabilities);
    ray_traced_texture = std::make_unique<Texture>
    (
            device,
            surface_capabilities.currentExtent.width,
            surface_capabilities.currentExtent.height,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_FORMAT_R16G16B16A16_UNORM
    );
}

void RayTracedRenderer::createCameraUniformBuffer()
{
    camera_uniform_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(GlobalUBO),
            1,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    );
    camera_uniform_buffer->map();
}

void RayTracedRenderer::createLightIndicesBuffer()
{
    std::vector<uint32_t> light_indices;
    size_t i = 0;
    for (auto& [_, object] : rendered_objects)
    {
        light_indices.push_back(i);
        ++i;
    }
    number_of_lights = light_indices.size();
    uint32_t instances = std::max(number_of_lights, 1u);
    light_indices_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(uint32_t),
            instances,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    light_indices_buffer->writeWithStagingBuffer(light_indices.data());
}

void RayTracedRenderer::createDescriptors()
{
    descriptor_pool = DescriptorPool::Builder(device)
            .setMaxSets(2)
            .addPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4)
            .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textures.size())
            .build();

    descriptor_set_layout = DescriptorSetLayout::Builder(device)
            .addBinding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
            .addBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, textures.size())
            .build();

    VkWriteDescriptorSetAccelerationStructureKHR acceleration_structure_descriptor_info{};
    acceleration_structure_descriptor_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    acceleration_structure_descriptor_info.accelerationStructureCount = 1;
    acceleration_structure_descriptor_info.pAccelerationStructures = &tlas.acceleration_structure;

    auto camera_buffer_descriptor_info = camera_uniform_buffer->descriptorInfo();
    auto object_descriptions_buffer_descriptor_info = object_descriptions_buffer->descriptorInfo();
    auto light_indices_descriptor_info = light_indices_buffer->descriptorInfo();
    auto material_descriptions_descriptor_info = material_descriptions_buffer->descriptorInfo();

    VkDescriptorImageInfo rayTraceImageDescriptorInfo = {
            .sampler = VK_NULL_HANDLE,
            .imageView = ray_traced_texture->getImageView(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    std::vector<VkDescriptorImageInfo> texture_descriptor_infos;
    texture_descriptor_infos.reserve(textures.size());
    for (auto& texture : textures)
    {
        VkDescriptorImageInfo texture_descriptor_info{};
        texture_descriptor_info.sampler = texture->getSampler();
        texture_descriptor_info.imageView = texture->getImageView();
        texture_descriptor_info.imageLayout = texture->getImageLayout();
        texture_descriptor_infos.emplace_back(texture_descriptor_info);
    }

    DescriptorWriter(*descriptor_set_layout, *descriptor_pool)
            .writeAccelerationStructure(0, &acceleration_structure_descriptor_info)
            .writeImage(1, &rayTraceImageDescriptorInfo)
            .writeBuffer(2, &camera_buffer_descriptor_info)
            .writeBuffer(3, &object_descriptions_buffer_descriptor_info)
            .writeBuffer(4, &light_indices_descriptor_info)
            .writeBuffer(5, &material_descriptions_descriptor_info)
            .writeImage(6, texture_descriptor_infos.data(), texture_descriptor_infos.size())
            .build(descriptor_set_handle);
}

void RayTracedRenderer::createRayTracingPipeline()
{
    ray_tracing_pipeline = RayTracingPipelineBuilder(device, ray_tracing_properties)
            .addRayGenerationStage(std::make_unique<ShaderModule>(device, "raytrace", VK_SHADER_STAGE_RAYGEN_BIT_KHR))
            .addMissStage(std::make_unique<ShaderModule>(device, "raytrace", VK_SHADER_STAGE_MISS_BIT_KHR))
            .addMissStage(std::make_unique<ShaderModule>(device, "raytrace_shadow", VK_SHADER_STAGE_MISS_BIT_KHR))
            .addHitGroup(
                    std::make_unique<ShaderModule>(device, "raytrace_lambertian", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
                    std::make_unique<ShaderModule>(device, "raytrace_occlusion", VK_SHADER_STAGE_ANY_HIT_BIT_KHR))
//            .addHitGroup(
//                    std::make_unique<ShaderModule>(device, "raytrace_light", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
//                    std::make_unique<ShaderModule>(device, "raytrace_occlusion", VK_SHADER_STAGE_ANY_HIT_BIT_KHR))
//            .addHitGroup(
//                    std::make_unique<ShaderModule>(device, "raytrace_specular", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
//                    std::make_unique<ShaderModule>(device, "raytrace_occlusion", VK_SHADER_STAGE_ANY_HIT_BIT_KHR))
            .addDescriptorSetLayout(descriptor_set_layout->getDescriptorSetLayout())
            .setMaxRecursionDepth(2)
            .build();
}

RayTracedRenderer::~RayTracedRenderer() noexcept
{
    pvkDestroyAccelerationStructureKHR(device.getDevice(), tlas.acceleration_structure, VulkanDefines::NO_CALLBACK);
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
    CameraUBO camera_ubo{};
    camera_ubo.camera_view = frame_info.camera_view_matrix;
    camera_ubo.camera_proj = frame_info.camera_projection_matrix;
    camera_uniform_buffer->writeToBuffer(&camera_ubo);
    camera_uniform_buffer->flush();
    ray_tracing_pipeline->bindDescriptorSets(frame_info.command_buffer, {descriptor_set_handle});
}

void RayTracedRenderer::updateRayPushConstant(FrameInfo& frame_info)
{
    PushConstantRay push_constant_ray{};
    push_constant_ray.time = std::chrono::system_clock::now().time_since_epoch().count();
    current_number_of_frames = frame_info.need_to_refresh_generated_image ? 0 : current_number_of_frames;
    push_constant_ray.frames = current_number_of_frames;
    push_constant_ray.number_of_lights = number_of_lights;
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