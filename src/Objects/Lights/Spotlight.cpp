#include "Spotlight.h"

Spotlight::Spotlight(Scene* parent_scene, std::shared_ptr<MaterialAsset> material,
        std::shared_ptr<RawModel> model_data, const glm::vec3& position,
        const glm::vec3& rotation, float scale, const glm::vec4& light_direction, const glm::vec3& light_color, float cutoff_angle)
    : Light(parent_scene, std::move(material), std::move(model_data), position, rotation, scale, light_direction,
            light_color, {1, 0, 0}, cutoff_angle) {}