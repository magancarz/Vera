#include "VulkanHelper.h"

const char* VulkanHelper::getResultString(VkResult result)
{
    const char* result_string = "unknown";

#define STR(a)                                                                                                         \
  case a:                                                                                                              \
    result_string = #a;                                                                                                \
    break;

    switch(result)
    {
        STR(VK_SUCCESS);
        STR(VK_NOT_READY);
        STR(VK_TIMEOUT);
        STR(VK_EVENT_SET);
        STR(VK_EVENT_RESET);
        STR(VK_INCOMPLETE);
        STR(VK_ERROR_OUT_OF_HOST_MEMORY);
        STR(VK_ERROR_OUT_OF_DEVICE_MEMORY);
        STR(VK_ERROR_INITIALIZATION_FAILED);
        STR(VK_ERROR_DEVICE_LOST);
        STR(VK_ERROR_MEMORY_MAP_FAILED);
        STR(VK_ERROR_LAYER_NOT_PRESENT);
        STR(VK_ERROR_EXTENSION_NOT_PRESENT);
        STR(VK_ERROR_FEATURE_NOT_PRESENT);
        STR(VK_ERROR_INCOMPATIBLE_DRIVER);
        STR(VK_ERROR_TOO_MANY_OBJECTS);
        STR(VK_ERROR_FORMAT_NOT_SUPPORTED);
        STR(VK_ERROR_FRAGMENTED_POOL);
        STR(VK_ERROR_OUT_OF_POOL_MEMORY);
        STR(VK_ERROR_INVALID_EXTERNAL_HANDLE);
        STR(VK_ERROR_SURFACE_LOST_KHR);
        STR(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR);
        STR(VK_SUBOPTIMAL_KHR);
        STR(VK_ERROR_OUT_OF_DATE_KHR);
        STR(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR);
        STR(VK_ERROR_VALIDATION_FAILED_EXT);
        STR(VK_ERROR_INVALID_SHADER_NV);
        STR(VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
        STR(VK_ERROR_FRAGMENTATION_EXT);
        STR(VK_ERROR_NOT_PERMITTED_EXT);
        STR(VK_ERROR_INVALID_DEVICE_ADDRESS_EXT);
        STR(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
    }
#undef STR
    return result_string;
}

bool VulkanHelper::checkResult(VkResult result, const char* file, int32_t line)
{
    if(result == VK_SUCCESS)
    {
        return false;
    }

    if(result < 0)
    {
        printf("%s(%d): Vulkan Error : %s\n", file, line, getResultString(result));
        assert(!"Critical Vulkan Error");

        return true;
    }

    return false;
}

bool VulkanHelper::checkResult(VkResult result, const char* message)
{
    if(result == VK_SUCCESS)
    {
        return false;
    }

    if(result < 0)
    {
        if(message)
        {
            printf("VkResult %d - %s - %s\n", result, getResultString(result), message);
        }
        else
        {
            printf("VkResult %d - %s\n", result, getResultString(result));
        }
        assert(!"Critical Vulkan Error");
        return true;
    }

    return false;
}

void VulkanHelper::loadExtensionsFunctions(VkDevice device)
{
    pvkCreateRayTracingPipelinesKHR =
            (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(
                    device, "vkCreateRayTracingPipelinesKHR");

    pvkGetAccelerationStructureBuildSizesKHR =
            (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(
                    device, "vkGetAccelerationStructureBuildSizesKHR");

    pvkCreateAccelerationStructureKHR =
            (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(
                    device, "vkCreateAccelerationStructureKHR");

    pvkDestroyAccelerationStructureKHR =
            (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(
                    device, "vkDestroyAccelerationStructureKHR");

    pvkGetAccelerationStructureDeviceAddressKHR =
            (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(
                    device, "vkGetAccelerationStructureDeviceAddressKHR");

    pvkCmdBuildAccelerationStructuresKHR =
            (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(
                    device, "vkCmdBuildAccelerationStructuresKHR");

    pvkGetRayTracingShaderGroupHandlesKHR =
            (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(
                    device, "vkGetRayTracingShaderGroupHandlesKHR");

    pvkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR");
}

VkTransformMatrixKHR VulkanHelper::mat4ToVkTransformMatrixKHR(glm::mat4 mat)
{
    mat = glm::transpose(mat);
    VkTransformMatrixKHR transform_matrix =
    {
            mat[0][0], mat[0][1], mat[0][2], mat[0][3],
            mat[1][0], mat[1][1], mat[1][2], mat[1][3],
            mat[2][0], mat[2][1], mat[2][2], mat[2][3]
    };

    return transform_matrix;
}