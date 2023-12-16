#pragma once

#include "EditorCommand.h"

class ToggleLiveRayTracingCommand : public EditorCommand
{
public:
    void execute(Editor* editor) override;
};