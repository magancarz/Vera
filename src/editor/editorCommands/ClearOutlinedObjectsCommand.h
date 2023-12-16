#pragma once

#include "EditorCommand.h"

class ClearOutlinedObjectsCommand : public EditorCommand
{
public:
    void execute(Editor* editor) override;
};