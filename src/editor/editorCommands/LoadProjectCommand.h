#pragma once

#include <string>

#include "EditorCommand.h"

class LoadProjectCommand : public EditorCommand
{
public:
    LoadProjectCommand(std::string project_name);

    void execute(Editor* editor) override;

private:
    std::string project_name;
};