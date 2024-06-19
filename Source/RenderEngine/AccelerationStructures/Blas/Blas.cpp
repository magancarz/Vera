#include "Blas.h"

#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "Assets/AssetManager.h"

Blas::Blas(
    VulkanHandler& device,
    MemoryAllocator& memory_allocator)
    : device{device}, memory_allocator{memory_allocator} {}

BlasInstance Blas::createBlasInstance(const glm::mat4& transform) const
{
    BlasInstance blas_instance{};
    blas_instance.bottom_level_acceleration_structure_instance.transform = VulkanHelper::mat4ToVkTransformMatrixKHR(transform);
    blas_instance.bottom_level_acceleration_structure_instance.mask = 0xFF;
    blas_instance.bottom_level_acceleration_structure_instance.instanceShaderBindingTableRecordOffset = 0;
    blas_instance.bottom_level_acceleration_structure_instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    blas_instance.bottom_level_acceleration_structure_instance.accelerationStructureReference = blas.getBuffer().getBufferDeviceAddress();

    BufferInfo bottom_level_geometry_instance_buffer_info{};
    bottom_level_geometry_instance_buffer_info.instance_size = sizeof(VkAccelerationStructureInstanceKHR);
    bottom_level_geometry_instance_buffer_info.instance_count = 1;
    bottom_level_geometry_instance_buffer_info.usage_flags =
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bottom_level_geometry_instance_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    bottom_level_geometry_instance_buffer_info.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    blas_instance.bottom_level_geometry_instance_buffer = memory_allocator.createBuffer(bottom_level_geometry_instance_buffer_info);
    blas_instance.bottom_level_geometry_instance_buffer->map();
    blas_instance.bottom_level_geometry_instance_buffer->writeToBuffer(&blas_instance.bottom_level_acceleration_structure_instance);
    blas_instance.bottom_level_geometry_instance_buffer->unmap();

    blas_instance.bottom_level_geometry_instance_device_address = blas_instance.bottom_level_geometry_instance_buffer->getBufferDeviceAddress();

    return blas_instance;
}

void Blas::update()
{
    BlasBuilder::updateBottomLevelAccelerationStructures(
        device, memory_allocator, {blas.getHandle()}, {blas_input}, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}
