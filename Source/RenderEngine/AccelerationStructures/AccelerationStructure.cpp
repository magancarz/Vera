#include "AccelerationStructure.h"

#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

AccelerationStructure::AccelerationStructure(Device& logical_device, VkAccelerationStructureKHR handle, std::unique_ptr<Buffer> buffer)
    : logical_device{logical_device}, handle{handle}, buffer{std::move(buffer)} {}

AccelerationStructure::~AccelerationStructure()
{
    destroyAccelerationStructureIfNeeded();
}

void AccelerationStructure::destroyAccelerationStructureIfNeeded()
{
    if (handle != VK_NULL_HANDLE)
    {
        pvkDestroyAccelerationStructureKHR(logical_device.getDevice(), handle, VulkanDefines::NO_CALLBACK);
    }
}

AccelerationStructure::AccelerationStructure(AccelerationStructure&& other) noexcept
    : logical_device{other.logical_device}, handle{other.handle}, buffer{std::move(other.buffer)}
{
    other.handle = VK_NULL_HANDLE;
}

AccelerationStructure& AccelerationStructure::operator=(AccelerationStructure&& other) noexcept
{
    destroyAccelerationStructureIfNeeded();

    handle = other.handle;
    other.handle = VK_NULL_HANDLE;
    buffer = std::move(other.buffer);

    return *this;
}
