#pragma once

#include "GUIElement.h"

class SceneOutlinerGUIElement : public GUIElement
{
public:
	virtual std::vector<std::shared_ptr<EditorCommand>> renderGUIElement(const EditorInfo& editor_info) override;

private:
    float ray_tracing_config_input_size = 6.f;
    bool is_editor_window_visible = true;
};