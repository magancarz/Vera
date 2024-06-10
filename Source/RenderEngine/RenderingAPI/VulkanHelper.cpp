#include "VulkanHelper.h"

void VulkanHelper::loadExtensionsFunctions(VkDevice device)
{
    pvkCreateRayTracingPipelinesKHR =
            reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(vkGetDeviceProcAddr(
                    device, "vkCreateRayTracingPipelinesKHR"));

    pvkGetAccelerationStructureBuildSizesKHR =
            reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(
                    device, "vkGetAccelerationStructureBuildSizesKHR"));

    pvkCreateAccelerationStructureKHR =
            reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(
                    device, "vkCreateAccelerationStructureKHR"));

    pvkDestroyAccelerationStructureKHR =
            reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(
                    device, "vkDestroyAccelerationStructureKHR"));

    pvkGetAccelerationStructureDeviceAddressKHR =
            reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(
                    device, "vkGetAccelerationStructureDeviceAddressKHR"));

    pvkCmdBuildAccelerationStructuresKHR =
            reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(
                    device, "vkCmdBuildAccelerationStructuresKHR"));

    pvkGetRayTracingShaderGroupHandlesKHR =
            reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(vkGetDeviceProcAddr(
                    device, "vkGetRayTracingShaderGroupHandlesKHR"));

    pvkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR"));

    pvkCmdWriteAccelerationStructuresPropertiesKHR = reinterpret_cast<PFN_vkCmdWriteAccelerationStructuresPropertiesKHR>(vkGetDeviceProcAddr(device, "vkCmdWriteAccelerationStructuresPropertiesKHR"));
    pvkCmdCopyAccelerationStructureKHR = reinterpret_cast<PFN_vkCmdCopyAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkCmdCopyAccelerationStructureKHR"));
}

VkTransformMatrixKHR VulkanHelper::mat4ToVkTransformMatrixKHR(glm::mat4 mat)
{
    mat = glm::transpose(mat);
    VkTransformMatrixKHR out_matrix;
    memcpy(&out_matrix, &mat, sizeof(VkTransformMatrixKHR));
    return out_matrix;
}