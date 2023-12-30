#pragma once

#include <memory>

#include "EditorCommand.h"
#include "Objects/Lights/LightCreators/LightCreator.h"

struct RawModel;

class CreateLightCommand : public EditorCommand
{
public:
    CreateLightCommand(std::shared_ptr<LightCreator> light_creator);

    void execute(Editor* editor) override;

private:
    std::shared_ptr<LightCreator> light_creator;
};
