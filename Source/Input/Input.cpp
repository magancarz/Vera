#include "Input.h"

#include <ranges>
#include <stdexcept>

#include "imgui/imgui.h"

void Input::initializeInput()
{
    for (const auto i : std::views::iota(0, NUM_KEYS))
    {
        key_down[i] = false;
    }
}

bool Input::isKeyDown(unsigned int key_code)
{
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return false;
    if (key_code >= NUM_KEYS)
        return false;
    return key_down[key_code];
}

void Input::set_key_down(unsigned int key_code, bool value)
{
    if (key_code >= NUM_KEYS)
    {
        throw std::runtime_error("Warning: tried to set key code bigger than NUM_KEYS value.\n");
    }
    key_down[key_code] = value;
}

void Input::set_left_mouse_button_down(const bool value)
{
    left_mouse_button_down = value;
}

void Input::set_right_mouse_button_down(const bool value)
{
    right_mouse_button_down = value;
}

bool Input::isLeftMouseButtonDown()
{
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return false;
    return left_mouse_button_down;
}

bool Input::isRightMouseButtonDown()
{
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return false;
    return right_mouse_button_down;
}
