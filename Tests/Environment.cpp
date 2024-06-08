#include "Environment.h"

#include <Mocks/MockLogger.h>
#include "Editor/Window/GLFWWindow.h"
#include "Editor/Window/WindowSystem.h"
#include "Logs/LogSystem.h"
#include "Memory/Vulkan/VulkanMemoryAllocator.h"

void Environment::SetUp()
{
    LogSystem::initialize(std::make_unique<MockLogger>());
    WindowSystem::initialize(std::make_unique<GLFWWindow>());
    vulkan_handler = std::make_unique<VulkanHandler>();
    memory_allocator = std::make_unique<VulkanMemoryAllocator>(*vulkan_handler);
}

void Environment::TearDown()
{
    memory_allocator.reset();
    vulkan_handler.reset();
    WindowSystem::initialize(nullptr);
    LogSystem::initialize(nullptr);
}