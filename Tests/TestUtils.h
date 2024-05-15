#pragma once

#include "gtest/gtest.h"

#include <glm/glm.hpp>

#include <string>

class TestUtils
{
public:
    static void deleteFileIfExists(const std::string& location);
    static bool fileExists(const std::string& file_location);
    static std::string loadFileToString(const std::string& file_location);

    static void printMatrix(const glm::mat4& matrix);

    template <typename T>
    static void assertTwoValuesAreEqual(T first_value, T second_value, double precision = 0.000001)
    {
        EXPECT_TRUE(abs(second_value - first_value) < precision);
    }

    static void assertTwoMatricesAreEqual(const glm::mat4& first_matrix, const glm::mat4& second_matrix);
};
