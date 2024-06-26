#pragma once

#include <Environment.h>

#include "gtest/gtest.h"
#include "Project/ObjectInfo.h"

#include <glm/glm.hpp>

#include <string>
#include <bit>
#include <Memory/Buffer.h>

class TestUtils
{
public:
    static void deleteFileIfExists(const std::string& location);
    static bool fileExists(const std::string& file_location);
    static std::string loadFileToString(const std::string& file_location);

    static void printMatrix(const glm::mat4& matrix);
    static void printVector(const glm::vec3& vector);

    template <typename T>
    static void expectTwoValuesToBeEqual(T first_value, T second_value, double precision = 0.000001)
    {
        EXPECT_TRUE(abs(second_value - first_value) < precision);
    }

    static void expectTwoVectorsToBeEqual(const glm::vec3& actual_vector, const glm::vec3& expected_vector);
    static void expectTwoMatricesToBeEqual(const glm::mat4& actual_matrix, const glm::mat4& expected_matrix);
    static void expectTwoMatricesToBeEqual(const VkTransformMatrixKHR& actual_matrix, const VkTransformMatrixKHR& expected_matrix);

    static ObjectInfo createDummyObjectInfo(
            std::string object_name,
            const glm::vec3& position = {0, 0, 0},
            const glm::vec3& rotation = {0, 0, 0},
            float scale = 0.f);

    static float randomFloat();
    static float randomFloat(float min, float max);
    static glm::mat4 randomTransform();

    template <typename T>
    static void expectBufferHasEqualData(const Buffer& buffer, const std::vector<T>& expected_data)
    {
        BufferInfo data_buffer_info{};
        data_buffer_info.instance_size = sizeof(T);
        data_buffer_info.instance_count = static_cast<uint32_t>(expected_data.size());
        data_buffer_info.usage_flags = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        data_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        data_buffer_info.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        auto data_buffer = TestsEnvironment::memoryAllocator().createBuffer(data_buffer_info);
        data_buffer->copyFrom(buffer);

        data_buffer->map();
        auto actual_data = std::bit_cast<T*>(data_buffer->getMappedMemory());
        for (size_t i = 0; i < expected_data.size(); ++i)
        {
            EXPECT_EQ(actual_data[i], expected_data[i]);
        }
    }

    static void failIfVulkanValidationLayersErrorsWerePresent();
};
