#pragma once

class InputManager
{
public:
    virtual ~InputManager() = default;

    virtual bool isKeyPressed(int key_mapping) = 0;
};