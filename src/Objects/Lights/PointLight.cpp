#include "PointLight.h"

PointLight::PointLight(Scene* parent_scene, std::shared_ptr<MaterialAsset> material, std::shared_ptr<RawModel> model_data,
        const glm::vec3& position, const glm::vec3& rotation, float scale, const glm::vec3& light_color, const glm::vec3& attenuation)
    : Light(parent_scene, std::move(material), std::move(model_data), position, rotation, scale, {0, -1, 0, 1}, light_color, attenuation) {}