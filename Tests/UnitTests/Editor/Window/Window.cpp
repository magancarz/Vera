#include "gtest/gtest.h"

#include <Editor/Window/WindowSystem.h>

TEST(WindowTests, shouldReturnCorrectAspect)
{
    // given
    const Window& window = WindowSystem::get();

    // when
    float aspect = window.getAspect();

    // then
    VkExtent2D extent = window.getExtent();
    EXPECT_EQ(aspect, static_cast<float>(extent.width) / extent.height);
}