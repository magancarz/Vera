#pragma once

#include "Material.h"

class MaterialBuilder
{
public:
    explicit MaterialBuilder(VulkanFacade& device);

    MaterialBuilder& lambertian();
    MaterialBuilder& color(const glm::vec3& color);

    MaterialBuilder& specular();
    MaterialBuilder& fuzziness(float value);

    MaterialBuilder& light();
    MaterialBuilder& brightness(unsigned int value);

    MaterialBuilder& name(std::string name);

    std::shared_ptr<Material> build();

private:
    VulkanFacade& device;

    MaterialInfo material_info{};
    std::string material_name{"material"};
    uint32_t material_hit_group_index{0};
};
