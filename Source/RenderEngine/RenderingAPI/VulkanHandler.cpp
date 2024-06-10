#include "VulkanHandler.h"

#include "VulkanHelper.h"

VulkanHandler::VulkanHandler()
{
    VulkanHelper::loadExtensionsFunctions(device.getDevice());
}