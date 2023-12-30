#include "LightCreator.h"

LightCreator::LightCreator(std::string light_type_name)
    : light_type_name{std::move(light_type_name)} {}

std::string LightCreator::getLightTypeName() const
{
    return light_type_name;
}

bool LightCreator::isPrefixValid(const std::string& light_info, const std::string& prefix)
{
    std::stringstream iss(light_info);
    std::string light_info_metadata, in_prefix;
    iss >> light_info_metadata;
    iss >> in_prefix;

    return in_prefix == prefix;
}