#pragma once

#include <map>
#include <memory>
#include <vector>

#include "GL/glew.h"

#include "Shaders/ShaderProgram.h"
#include "RenderEngine/Structs/LightInfo.h"
#include "RenderEngine/Structs/ProjectionMatrices.h"
#include "UniformBuffer.h"

struct RawModel;
class Camera;
class Light;

class SceneObjectsShader : public ShaderProgram
{
public:
    SceneObjectsShader();
    ~SceneObjectsShader() override = default;

    void connectTextureUnits() const;
    void getAllUniformLocations() override;

    void loadTransformationMatrix(const glm::mat4& matrix) const;
    void loadViewAndProjectionMatrices(const std::shared_ptr<Camera>& camera) const;
    void loadReflectivity(float reflectivity) const;

    void loadLights(const std::vector<std::weak_ptr<Light>>& lights) const;

    inline static constexpr int MAX_LIGHTS = 4;

protected:
    SceneObjectsShader(const std::string& vertex_file, const std::string& fragment_file);
    SceneObjectsShader(const std::string& vertex_file, const std::string& geometry_file, const std::string& fragment_file);

private:
    void prepareUniformBuffers();

    const std::string LIGHT_INFO_UNIFORM_BLOCK_NAME = "LightInfos";
    UniformBuffer<LightInfo> light_info_uniform_buffer;
    const std::string PROJECTION_MATRICES_UNIFORM_BLOCK_NAME = "ProjectionMatrices";
    UniformBuffer<ProjectionMatrices> projection_matrices_uniform_buffer;

    int location_transformation_matrix;
    int location_reflectivity;

    int location_model_texture;
    int location_shadow_map;
};
