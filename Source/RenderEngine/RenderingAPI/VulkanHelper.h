#pragma once

#include <cstdint>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>

inline PFN_vkCreateRayTracingPipelinesKHR pvkCreateRayTracingPipelinesKHR;
inline PFN_vkGetAccelerationStructureBuildSizesKHR pvkGetAccelerationStructureBuildSizesKHR;
inline PFN_vkCreateAccelerationStructureKHR pvkCreateAccelerationStructureKHR;
inline PFN_vkDestroyAccelerationStructureKHR pvkDestroyAccelerationStructureKHR;
inline PFN_vkGetAccelerationStructureDeviceAddressKHR pvkGetAccelerationStructureDeviceAddressKHR;
inline PFN_vkCmdBuildAccelerationStructuresKHR pvkCmdBuildAccelerationStructuresKHR;
inline PFN_vkGetRayTracingShaderGroupHandlesKHR pvkGetRayTracingShaderGroupHandlesKHR;
inline PFN_vkCmdTraceRaysKHR pvkCmdTraceRaysKHR;
inline PFN_vkCmdWriteAccelerationStructuresPropertiesKHR pvkCmdWriteAccelerationStructuresPropertiesKHR;
inline PFN_vkCmdCopyAccelerationStructureKHR pvkCmdCopyAccelerationStructureKHR;

class VulkanHelper
{
public:
    static void loadExtensionsFunctions(VkDevice device);
    static VkTransformMatrixKHR mat4ToVkTransformMatrixKHR(glm::mat4 mat);

    template <class integral>
    static constexpr integral align_up(integral x, size_t a) noexcept
    {
        return integral((x + (integral(a) - 1)) & ~integral(a - 1));
    }
};
