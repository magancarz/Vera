#pragma once

#include "GUIComponent.h"

class SunSettingsComponent : public GUIComponent
{
public:
    SunSettingsComponent();

    void update(FrameInfo& frame_info) override;

    inline static const char* const DISPLAY_NAME{"Sun Settings"};

private:
    void drawSliderElements();
    void calculateSunPosition(FrameInfo& frame_info) const;
    void updateFrameInfo(FrameInfo& frame_info) const;
    void updatePreviousValues();

    float previous_sun_yaw_angle{30.f};
    float sun_yaw_angle{30.f};

    float previous_sun_pitch_angle{30.f};
    float sun_pitch_angle{30.f};

    float previous_weather{0.05f};
    float weather{0.05f};
};
