#pragma once

#include <map>
#include <memory>
#include <vector>

#include "Shaders/ShaderProgram.h"
#include "GL/glew.h"

struct RawModel;
class Camera;
class Light;

class ParallaxMappedSceneObjectsShader : public ShaderProgram
{
public:
    ParallaxMappedSceneObjectsShader();

    void loadTransformationMatrix(const glm::mat4& matrix) const;
    void loadViewMatrix(const std::shared_ptr<Camera>& camera) const;
    void loadProjectionMatrix(const std::shared_ptr<Camera>& camera) const;
    void loadLightsCount(size_t count);
    void loadReflectivity(float reflectivity) const;
    void loadHeightScale(float height_scale) const;
    void loadTextureIndexToCubeShadowMap(size_t cube_shadow_map_index, unsigned int texture_index);

    void connectTextureUnits(GLenum texture) const;
    void getAllUniformLocations() override;

    void loadLights(const std::vector<std::weak_ptr<Light>>& lights) const;

    inline static constexpr int MAX_LIGHTS = 4;

private:
    int location_transformation_matrix;
    int location_projection_matrix;
    int location_view_matrix;

    int location_model_texture;
    int location_normal_texture;
    int location_depth_texture;
    int location_shadow_cube_map_textures[MAX_LIGHTS];

    int location_lights_count;
    int location_light_position[MAX_LIGHTS];
    int location_light_color[MAX_LIGHTS];
    int location_attenuation[MAX_LIGHTS];

    int location_reflectivity;
    int location_height_scale;
};
