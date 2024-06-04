#include "SunSettingsComponent.h"

#include "imgui.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/transform.hpp"

SunSettingsComponent::SunSettingsComponent()
    : Component(DISPLAY_NAME) {}

void SunSettingsComponent::update(FrameInfo& frame_info)
{
    drawSliderElements();
    calculateSunPosition(frame_info);
    updateFrameInfo(frame_info);
    updatePreviousValues();
}

void SunSettingsComponent::drawSliderElements()
{
    ImGui::SliderFloat("Sun Yaw Angle", &sun_yaw_angle, 0.0f, 360.0f);
    ImGui::SliderFloat("Sun Pitch Angle", &sun_pitch_angle, 0.0f, 90.0f);
    ImGui::SliderFloat("Weather", &weather, 0.01f, 1.0f);
}

void SunSettingsComponent::calculateSunPosition(FrameInfo& frame_info) const
{
    glm::vec3 initial_sun_position{1.0f, 0.0f, 0.0f};
    glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(sun_yaw_angle), glm::vec3{0, 1, 0});
    rotation = glm::rotate(rotation, glm::radians(sun_pitch_angle), glm::vec3{0, 0, 1});
    initial_sun_position = rotation * glm::vec4{initial_sun_position, 0.0f};
    frame_info.sun_position = initial_sun_position;
}

void SunSettingsComponent::updateFrameInfo(FrameInfo& frame_info) const
{
    frame_info.weather = weather;
    frame_info.need_to_refresh_generated_image |= glm::abs(previous_sun_yaw_angle - sun_yaw_angle) > 0.001f
                                                  || glm::abs(previous_sun_pitch_angle - sun_pitch_angle) > 0.001f
                                                  || glm::abs(previous_weather - weather) > 0.001f;
}

void SunSettingsComponent::updatePreviousValues()
{
    previous_sun_yaw_angle = sun_yaw_angle;
    previous_sun_pitch_angle = sun_pitch_angle;
    previous_weather = weather;
}