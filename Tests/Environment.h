#pragma once

#include <TestLogger.h>

#include "gtest/gtest.h"
#include "Memory/MemoryAllocator.h"
#include "RenderEngine/RenderingAPI/VulkanHandler.h"

class TestsEnvironment : public ::testing::Environment
{
public:
    void SetUp() override;
    void TearDown() override;

    [[nodiscard]] static VulkanHandler& vulkanHandler() { return *vulkan_handler; }
    [[nodiscard]] static MemoryAllocator& memoryAllocator() { return *memory_allocator; }

    static void initializeTestsLogger();
    [[nodiscard]] static TestLogger& testLogger() { return *test_logger; }

private:
    void failAllTestsIfThereWereAnyVulkanValidationLayersErrorsDuringSetup();

    inline static std::unique_ptr<VulkanHandler> vulkan_handler;
    inline static std::unique_ptr<MemoryAllocator> memory_allocator;
    inline static TestLogger* test_logger;
};