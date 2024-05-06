#pragma once

#include "Component.h"

class SunSettingsComponent : public Component
{
public:
    SunSettingsComponent();

    void update(FrameInfo& frame_info) override;

    inline static const char* const DISPLAY_NAME{"Sun Settings"};

private:
    float previous_sun_yaw_angle{30.f};
    float sun_yaw_angle{30.f};

    float previous_sun_pitch_angle{30.f};
    float sun_pitch_angle{30.f};

    float previous_weather{0.05f};
    float weather{0.05f};
};
