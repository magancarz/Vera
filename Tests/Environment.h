#pragma once

#include "gtest/gtest.h"
#include "Memory/MemoryAllocator.h"
#include "RenderEngine/RenderingAPI/VulkanHandler.h"

class Environment : public ::testing::Environment
{
public:
    void SetUp() override;
    void TearDown() override;

    [[nodiscard]] static VulkanHandler& vulkanHandler() { return *vulkan_handler; }
    [[nodiscard]] static MemoryAllocator& memoryAllocator() { return *memory_allocator; }

private:
    inline static std::unique_ptr<VulkanHandler> vulkan_handler;
    inline static std::unique_ptr<MemoryAllocator> memory_allocator;
};