#pragma once

class Editor;

class EditorCommand
{
public:
	virtual ~EditorCommand() = default;

	virtual void execute(Editor* editor) = 0;
};
