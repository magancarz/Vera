#pragma once

#include "Objects/Lights/Light.h"

class Spotlight : public Light
{
public:
    Spotlight(Scene* parent_scene, std::shared_ptr<MaterialAsset> material, std::shared_ptr<RawModel> model_data,
        const glm::vec3& position, const glm::vec3& rotation, float scale, const glm::vec4& light_direction,
        const glm::vec3& light_color, float cutoff_angle);
};