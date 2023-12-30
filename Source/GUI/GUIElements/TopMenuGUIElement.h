#pragma once

#include <string>

#include "GUIElement.h"

class TopMenuGUIElement : public GUIElement
{
public:
	virtual std::vector<std::shared_ptr<EditorCommand>> renderGUIElement(const EditorInfo& editor_info);

private:
	bool show_loading_project_window{false};
	bool show_saving_project_window{false};
	std::vector<std::string> available_project_files;
	std::string project_name;
	int selected_project_idx{-1};
};