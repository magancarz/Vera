#pragma once

#include <gmock/gmock.h>

#include "RenderEngine/Models/Model.h"

class MockModel : public Model
{
public:
    explicit MockModel(std::string model_name);

    MOCK_METHOD(std::vector<std::string>, getRequiredMaterials, (), (const, override));
};