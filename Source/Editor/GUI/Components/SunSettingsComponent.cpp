#include "SunSettingsComponent.h"

#include "imgui.h"

SunSettingsComponent::SunSettingsComponent()
    : Component(DISPLAY_NAME) {}

void SunSettingsComponent::update(FrameInfo& frame_info)
{
    ImGui::SliderFloat("Sun Yaw Angle", &sun_yaw_angle, 0.0f, 360.0f);
    ImGui::SliderFloat("Sun Pitch Angle", &sun_pitch_angle, 0.0f, 90.0f);
    glm::vec3 initial_sun_position{1.0f, 0.0f, 0.0f};
    glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(sun_yaw_angle), glm::vec3{0, 1, 0});
    rotation = glm::rotate(rotation, glm::radians(sun_pitch_angle), glm::vec3{0, 0, 1});
    initial_sun_position = rotation * glm::vec4{initial_sun_position, 0.0f};
    frame_info.sun_position = initial_sun_position;

    ImGui::SliderFloat("Weather", &weather, 0.01f, 1.0f);

    frame_info.weather = weather;

    frame_info.player_moved |= abs(previous_sun_yaw_angle - sun_yaw_angle) > 0.001f
                               || abs(previous_sun_pitch_angle - sun_pitch_angle) > 0.001f
                               || abs(previous_weather - weather) > 0.001f;

    previous_sun_yaw_angle = sun_yaw_angle;
    previous_sun_pitch_angle = sun_pitch_angle;
    previous_weather = weather;
}