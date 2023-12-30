#pragma once

#include "EditorCommand.h"

class StopGeneratingImageCommand : public EditorCommand
{
public:
	StopGeneratingImageCommand() = default;

    void execute(Editor* editor) override;
};