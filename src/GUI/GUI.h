#pragma once

#include <memory>
#include <vector>
#include <string>
#include <optional>

#include "GUIElements/GUIElement.h"
#include "glm/glm.hpp"

class EditorCommand;
class Object;
struct EditorInfo;

class GUI
{
public:
    GUI();

    [[nodiscard]] std::vector<std::shared_ptr<EditorCommand>> renderGUI(const EditorInfo& editor_info);

    static bool drawInputFieldForFloat(float* value, const std::string& name, float field_size = INPUT_FIELD_SIZE / 3);
    static std::optional<glm::vec3> drawInputFieldForVector3(glm::vec3& vector, const std::string& name, float field_size = INPUT_FIELD_SIZE);
    static std::optional<glm::vec4> drawInputFieldForVector4(glm::vec4& vector, const std::string& name, float field_size = INPUT_FIELD_SIZE);

    inline static constexpr float INPUT_FIELD_SIZE = 18.f;

private:
    std::vector<std::unique_ptr<GUIElement>> gui_elements;
};
