#pragma once

#include "Models/RawModel.h"

class PostProcessingRenderer
{
public:
    virtual ~PostProcessingRenderer() = default;

    virtual void apply(const utils::Texture& in_out_hdr_color_buffer) = 0;
};
