#pragma once

#include <map>
#include <memory>
#include <vector>

#include "../shaders/ShaderProgram.h"

struct RawModel;
class Camera;
class Object;

class StaticShader : public ShaderProgram
{
public:
    StaticShader();

    void loadTransformationMatrix(const glm::mat4& matrix) const;
    void loadProjectionMatrix(const std::shared_ptr<Camera>& camera) const;
    void loadViewMatrix(const std::shared_ptr<Camera>& camera) const;
    void connectTextureUnits() const;

    void bindAttributes() override;
    void getAllUniformLocations() override;

    void loadLights(const std::map<std::shared_ptr<RawModel>, std::vector<std::shared_ptr<Object>>>& entity_map) const;

private:
    int location_transformation_matrix;
    int location_projection_matrix;
    int location_view_matrix;
    int location_model_texture;

    inline static constexpr int MAX_LIGHTS = 4;
    int location_light_position[MAX_LIGHTS];
    int location_light_direction[MAX_LIGHTS];
    int location_light_color[MAX_LIGHTS];
    int location_attenuation[MAX_LIGHTS];
    int location_cutoff_angle[MAX_LIGHTS];
};
