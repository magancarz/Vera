#pragma once

#include <memory>
#include <vector>

class GUIElement
{
public:
	virtual ~GUIElement() = default;

	virtual std::vector<std::shared_ptr<class EditorCommand>> renderGUIElement(const struct EditorInfo& editor_info) = 0;
};
