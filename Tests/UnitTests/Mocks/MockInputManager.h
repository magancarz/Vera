#pragma once

#include <gmock/gmock.h>

#include "Input/InputManager.h"

class MockInputManager : public InputManager
{
public:
    MOCK_METHOD(bool, isKeyPressed, (int key_mapping), (override));
};