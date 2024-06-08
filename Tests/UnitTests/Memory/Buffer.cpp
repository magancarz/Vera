#include "gtest/gtest.h"

#include <Environment.h>
#include <TestUtils.h>

TEST(BufferTests, shouldMapBufferMemory)
{
    // given
    auto buffer = TestsEnvironment::memoryAllocator().createBuffer(
        sizeof(uint32_t), 22,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    // when
    buffer->map();

    // then
    EXPECT_TRUE(buffer->getMappedMemory() != nullptr);
}

TEST(BufferTests, shouldUnmapBufferMemory)
{
    // given
    auto buffer = TestsEnvironment::memoryAllocator().createBuffer(
        sizeof(uint32_t), 22,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    buffer->map();

    ASSERT_TRUE(buffer->getMappedMemory() != nullptr);

    // when
    buffer->unmap();

    // then
    EXPECT_TRUE(buffer->getMappedMemory() == nullptr);
}

TEST(BufferTests, shouldWriteToStagingBuffer)
{
    // given
    std::vector<uint8_t> data_to_write{2, 1, 3, 7};
    auto staging_buffer = TestsEnvironment::memoryAllocator().createStagingBuffer(
        sizeof(uint8_t), static_cast<uint32_t>(data_to_write.size()));

    // when
    staging_buffer->writeToBuffer(data_to_write.data());

    // then
    TestUtils::expectBufferHasEqualData(*staging_buffer, data_to_write);
}

TEST(BufferTests, shouldCopyFromStagingBuffer)
{
    // given
    std::vector<uint8_t> data{2, 1, 3, 7};
    auto staging_buffer = TestsEnvironment::memoryAllocator().createStagingBuffer(
        sizeof(uint8_t), static_cast<uint32_t>(data.size()), data.data());

    auto dest_buffer = TestsEnvironment::memoryAllocator().createBuffer(
        sizeof(uint8_t), static_cast<uint32_t>(data.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // when
    dest_buffer->copyFrom(*staging_buffer);

    // then
    TestUtils::expectBufferHasEqualData(*dest_buffer, data);
}

TEST(BufferTests, shouldReturnValidBufferHandle)
{
    // given
    auto buffer = TestsEnvironment::memoryAllocator().createBuffer(
        sizeof(uint8_t),
        static_cast<uint32_t>(22),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

    // when
    VkBuffer buffer_handle = buffer->getBuffer();

    // then
    EXPECT_TRUE(buffer_handle != VK_NULL_HANDLE);
}

TEST(BufferTests, shouldReturnCorrectBufferSize)
{
    // given
    uint32_t instance_size = sizeof(uint8_t);
    uint32_t instances_count = 22;
    auto buffer = TestsEnvironment::memoryAllocator().createBuffer(
        instance_size,
        instances_count,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    // when
    uint32_t buffer_size  = buffer->getSize();

    // then
    EXPECT_EQ(buffer_size, instance_size * instances_count);
}

TEST(BufferTests, shouldReturnValidDeviceAddress)
{
    // given
    auto buffer = TestsEnvironment::memoryAllocator().createBuffer(
        sizeof(uint8_t),
        static_cast<uint32_t>(22),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

    // when
    VkDeviceAddress device_address = buffer->getBufferDeviceAddress();

    // then
    EXPECT_TRUE(device_address != 0ULL);
}

TEST(BufferTests, shouldReturnValidDescriptorInfo)
{
    // given
    uint32_t instance_size = sizeof(uint8_t);
    uint32_t instances_count = 22;
    auto buffer = TestsEnvironment::memoryAllocator().createBuffer(
        instance_size,
        instances_count,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    // when
    VkDescriptorBufferInfo descriptor_buffer_info  = buffer->descriptorInfo();

    // then
    EXPECT_NE(descriptor_buffer_info.buffer, VK_NULL_HANDLE);
    EXPECT_EQ(descriptor_buffer_info.offset, 0);
    EXPECT_EQ(descriptor_buffer_info.range, VK_WHOLE_SIZE);
}