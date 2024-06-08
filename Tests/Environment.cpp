#include "Environment.h"

#include <Mocks/MockLogger.h>
#include "Editor/Window/GLFWWindow.h"
#include "Editor/Window/WindowSystem.h"
#include "Logs/LogSystem.h"
#include "Memory/Vulkan/VulkanMemoryAllocator.h"
#include "TestLogger.h"

void TestsEnvironment::SetUp()
{
    initializeTestsLogger();

    WindowSystem::initialize(std::make_unique<GLFWWindow>());
    vulkan_handler = std::make_unique<VulkanHandler>();
    memory_allocator = std::make_unique<VulkanMemoryAllocator>(*vulkan_handler);

    failAllTestsIfThereWereAnyVulkanValidationLayersErrorsDuringSetup();
}

void TestsEnvironment::initializeTestsLogger()
{
    auto new_test_logger = std::make_unique<TestLogger>();
    test_logger = new_test_logger.get();
    LogSystem::initialize(std::move(new_test_logger));
}

void TestsEnvironment::failAllTestsIfThereWereAnyVulkanValidationLayersErrorsDuringSetup() const
{
    if (test_logger->anyVulkanValidationLayersErrors())
    {
        GTEST_FATAL_FAILURE_("Vulkan validation layer messages occured during Vulkan setup!");
    }
}

void TestsEnvironment::TearDown()
{
    memory_allocator.reset();
    vulkan_handler.reset();
    WindowSystem::initialize(nullptr);
    LogSystem::initialize(nullptr);
}