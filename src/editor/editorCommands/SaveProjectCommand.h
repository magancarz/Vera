#pragma once

#include "EditorCommand.h"

class SaveProjectCommand : public EditorCommand
{
public:
    SaveProjectCommand() = default;

    void execute(Editor* editor) override;
};