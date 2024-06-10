#pragma once

#include <gmock/gmock.h>

#include "Input/InputManager.h"

class MockInputManager : public InputManager
{
public:
    MOCK_METHOD(bool, isKeyPressed, (KeyCode key_code), (override));
};