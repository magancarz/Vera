#pragma once

#include <memory>

#include "Memory/Buffer.h"

class AccelerationStructure
{
public:
    AccelerationStructure(Device& logical_device, VkAccelerationStructureKHR handle, std::unique_ptr<Buffer> buffer);
    ~AccelerationStructure();

    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure(AccelerationStructure&&) noexcept;
    AccelerationStructure& operator=(AccelerationStructure&&) noexcept;

    [[nodiscard]] VkAccelerationStructureKHR getHandle() const { return handle; }
    [[nodiscard]] const Buffer& getBuffer() const { return *buffer; }

private:
    Device& logical_device;
    VkAccelerationStructureKHR handle;
    std::unique_ptr<Buffer> buffer;

    void destroyAccelerationStructureIfNeeded();
};