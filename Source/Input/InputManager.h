#pragma once

#include "KeyCodes.h"

class InputManager
{
public:
    virtual ~InputManager() = default;

    virtual bool isKeyPressed(KeyCode key_mapping) = 0;
};