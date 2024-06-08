#pragma once

#include <gmock/gmock-function-mocker.h>

class MockGUIComponent : public GUIComponent
{
public:
    MockGUIComponent(std::string name) : GUIComponent(std::move(name)) {}

    MOCK_METHOD(void, update, (FrameInfo&), (override));
};
