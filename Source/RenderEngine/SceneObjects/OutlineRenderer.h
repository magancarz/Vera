#pragma once

#include "OutlineMarkShader.h"
#include "OutlineShader.h"

class OutlineRenderer
{
public:
    OutlineRenderer();



private:
    OutlineMarkShader outline_mark_shader;
    OutlineShader outline_shader;
};