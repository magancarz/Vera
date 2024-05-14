#pragma once

#include <filesystem>

class PathBuilder
{
public:
    explicit PathBuilder(std::filesystem::path  starting_path = std::filesystem::current_path());

    PathBuilder& append(const std::string& path);
    std::string build();

private:
    std::filesystem::path current_path;
};
