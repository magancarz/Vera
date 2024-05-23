#pragma once

class MockInputManager : public InputManager
{
public:
    MOCK_METHOD(bool, isKeyPressed, (int key_mapping), (override));
};