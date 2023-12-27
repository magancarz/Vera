#pragma once

#include "Objects/Lights/Light.h"

class LightCreator
{
public:
    LightCreator(std::string light_type_name);

    virtual bool apply(const std::string& light_info) = 0;
    virtual std::shared_ptr<Light> fromLightInfo(Scene* parent_scene, const std::string& light_info) = 0;
    virtual std::shared_ptr<Light> create(Scene* parent_scene) = 0;

    std::string getLightTypeName() const;

protected:
    bool isPrefixValid(const std::string& light_info, const std::string& prefix);

    std::string light_type_name;
};