#pragma once

#include <glm/glm.hpp>

#include <memory>
#include <vector>
#include <string>
#include <optional>

#include "GUIElements/GUIElement.h"

class GUI
{
public:
    GUI();

    void renderGUI();

private:
    std::vector<std::unique_ptr<GUIElement>> gui_elements;
};
