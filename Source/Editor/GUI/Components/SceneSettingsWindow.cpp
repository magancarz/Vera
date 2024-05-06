#include "SceneSettingsWindow.h"

#include "SunSettingsComponent.h"

SceneSettingsWindow::SceneSettingsWindow()
    : WindowComponent(DISPLAY_NAME)
{
    createChildComponents();
}

void SceneSettingsWindow::createChildComponents()
{
    addComponent(std::make_shared<SunSettingsComponent>());
}