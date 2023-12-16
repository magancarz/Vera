#pragma once

#include "editor/Editor.h"

class TinyRayTracingApp
{
public:
    int launch();
    void run();
    void close();

private:
    std::shared_ptr<Editor> editor;
};
