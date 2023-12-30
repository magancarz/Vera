#pragma once

#include <memory>

#include "EditorCommand.h"

struct RawModel;

class CreateObjectCommand : public EditorCommand
{
public:
	CreateObjectCommand(std::shared_ptr<RawModel> model);

    void execute(Editor* editor) override;

private:
    std::shared_ptr<RawModel> model;
};
