#pragma once

#include <memory>

#include "EditorCommand.h"

class Object;

class SetObjectToBeOutlinedCommand : public EditorCommand
{
public:
	SetObjectToBeOutlinedCommand(std::weak_ptr<Object> object_to_be_outlined);

    void execute(Editor* editor) override;

private:
	std::weak_ptr<Object> object_to_be_outlined;
};
