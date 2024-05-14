#pragma once

#include <string>

class TestUtils
{
public:
    static void deleteFileIfExists(const std::string& location);
    static bool fileExists(const std::string& file_location);
    static std::string loadFileToString(const std::string& file_location);
};
