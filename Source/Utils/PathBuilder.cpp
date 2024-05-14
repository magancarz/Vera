#include "PathBuilder.h"

#include <utility>

PathBuilder::PathBuilder(std::filesystem::path  starting_path)
    : current_path{std::move(starting_path)} {}

PathBuilder& PathBuilder::append(const std::string& path)
{
    current_path /= std::filesystem::path(path);
    return *this;
}

std::string PathBuilder::build()
{
    return current_path.generic_string();
}
