#include "TestUtils.h"

#include <filesystem>
#include <iostream>
#include <fstream>

void TestUtils::deleteFileIfExists(const std::string& location)
{
    if (fileExists(location))
    {
        std::filesystem::remove(location);
        std::cout << "File at location " << location << " removed successfully\n";
        return;
    }
}

bool TestUtils::fileExists(const std::string& file_location)
{
    std::filesystem::path canonical_path = std::filesystem::canonical(file_location);
    if (std::filesystem::exists(canonical_path))
    {
        return true;
    }

    std::cout << "Couldn't find any file at location " << file_location << "\n";
    return false;
}

std::string TestUtils::loadFileToString(const std::string& file_location)
{
    if (!fileExists(file_location))
    {
        return "";
    }

    std::ifstream file(file_location);
    if (!file.is_open()) {
        std::cerr << "Error while opening file " << file_location << "\n";
        std::cout << "Current working directory is " << std::filesystem::current_path() << "\n";
        return "";
    }

    std::string content;
    std::string line;
    while (std::getline(file, line))
    {
        content += line + "\n";
    }

    file.close();
    return content;
}

void TestUtils::printMatrix(const glm::mat4& matrix)
{
    printf("%f, %f, %f, %f\n", matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]);
    printf("%f, %f, %f, %f\n", matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]);
    printf("%f, %f, %f, %f\n", matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]);
    printf("%f, %f, %f, %f\n\n", matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]);
}

void TestUtils::assertTwoMatricesAreEqual(const glm::mat4& first_matrix, const glm::mat4& second_matrix)
{
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            assertTwoValuesAreEqual(first_matrix[i][j], second_matrix[i][j]);
        }
    }
}