#pragma once

#include <cstdint>
#include <vector>

struct TextureData
{
    std::string name;
    uint32_t width{0};
    uint32_t height{0};
    uint32_t number_of_channels{4};
    std::vector<unsigned char> data;
};