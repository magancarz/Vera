#pragma once

#include "Light.h"

class PointLight : public Light
{
public:
    PointLight(Scene* parent_scene, std::shared_ptr<MaterialAsset> material, std::shared_ptr<RawModel> model_data,
        const glm::vec3& position, const glm::vec3& rotation, float scale, const glm::vec3& light_color,
        const glm::vec3& attenuation);

    ~PointLight() override = default;
};