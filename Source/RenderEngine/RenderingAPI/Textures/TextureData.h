#pragma once

#include <string>

struct TextureData
{
    int width{}, height{};
    int number_of_channels{};
    int expected_number_of_channels{4};
    unsigned char* data;

    explicit TextureData(const std::string& texture_name);

    ~TextureData();

    TextureData(const TextureData&) = delete;
    TextureData& operator=(const TextureData&) = delete;
    TextureData(TextureData&&) = delete;
    TextureData& operator=(TextureData&&) = delete;

};
