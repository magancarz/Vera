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