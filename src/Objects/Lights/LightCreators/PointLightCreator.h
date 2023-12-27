#pragma once

#include "LightCreator.h"
#include "Objects/Lights/PointLight.h"

class PointLightCreator : public LightCreator
{
public:
    PointLightCreator();

    bool apply(const std::string &light_info) override;
    std::shared_ptr<Light> fromLightInfo(Scene* parent_scene, const std::string& light_info) override;
    std::shared_ptr<Light> create(Scene* parent_scene) override;
};