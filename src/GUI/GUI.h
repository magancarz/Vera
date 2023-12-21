#pragma once

#include <memory>
#include <vector>

#include "GUIElements/GUIElement.h"

class EditorCommand;
class Object;
struct EditorInfo;

class GUI
{
public:
    GUI();

    [[nodiscard]] std::vector<std::shared_ptr<EditorCommand>> renderGUI(const EditorInfo& editor_info);

private:
    std::vector<std::unique_ptr<GUIElement>> gui_elements;
};
