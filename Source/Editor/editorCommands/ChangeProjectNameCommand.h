#pragma once

#include <string>

#include "EditorCommand.h"

class ChangeProjectNameCommand : public EditorCommand
{
public:
    ChangeProjectNameCommand(std::string project_name);

    void execute(Editor* editor) override;

private:
    std::string project_name;
};