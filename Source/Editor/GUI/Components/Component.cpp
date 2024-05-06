#include "Component.h"

Component::Component(std::string name)
    : component_name{std::move(name)}
{
    printf("Creating component %s\n", component_name.c_str());
}