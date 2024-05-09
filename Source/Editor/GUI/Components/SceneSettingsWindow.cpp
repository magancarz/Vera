#include "SceneSettingsWindow.h"

#include "SunSettingsComponent.h"
#include "FramerateTextComponent.h"

SceneSettingsWindow::SceneSettingsWindow()
    : WindowComponent(DISPLAY_NAME)
{
    createChildComponents();
}

void SceneSettingsWindow::createChildComponents()
{
    addComponent(std::make_shared<FramerateTextComponent>());
    addComponent(std::make_shared<SunSettingsComponent>());
}