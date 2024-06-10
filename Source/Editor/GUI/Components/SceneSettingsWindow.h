#pragma once

#include "WindowComponent.h"

class SceneSettingsWindow : public WindowComponent
{
public:
    SceneSettingsWindow();

    inline static const char* const DISPLAY_NAME{"Scene Settings"};

private:
    void createChildComponents();
};
