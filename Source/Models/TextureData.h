#pragma once

struct TextureData
{
    ~TextureData();

    unsigned char* texture_data;
    int width, height;
};
