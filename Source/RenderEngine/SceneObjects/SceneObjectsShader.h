#pragma once

#include <map>
#include <memory>
#include <vector>

#include "GL/glew.h"

#include "Shaders/ShaderProgram.h"
#include "RenderEngine/Structs/LightInfo.h"
#include "RenderEngine/Structs/TransformationMatrices.h"
#include "UniformBuffer.h"
#include "Objects/TriangleMesh.h"

struct RawModel;
class Camera;
class Light;

class SceneObjectsShader : public ShaderProgram
{
public:
    SceneObjectsShader();
    ~SceneObjectsShader() override = default;

    void connectTextureUnits() override;
    void getAllUniformLocations() override;

    void loadReflectivity(float reflectivity);

protected:
    SceneObjectsShader(const std::string& vertex_file, const std::string& fragment_file);
    SceneObjectsShader(const std::string& vertex_file, const std::string& geometry_file, const std::string& fragment_file);

private:
    int location_reflectivity;
    int location_model_texture;
};
