#pragma once

#include <vector>
#include <memory>

#include "ObjectComponent.h"
#include "Assets/Model/MeshDescription.h"

class Mesh;
class Model;
class Material;

class MeshComponent : public ObjectComponent
{
public:
    explicit MeshComponent(Object& owner);

    void setMesh(Mesh* mesh);

    [[nodiscard]] std::string getMeshName() const { return name; }
    [[nodiscard]] std::vector<Model*> getModels() const { return models; }

    [[nodiscard]] std::vector<Material*> getMaterials() const { return materials; }
    [[nodiscard]] Material* findMaterial(const std::string& name) const;
    [[nodiscard]] std::vector<std::string> getRequiredMaterials() const;

    [[nodiscard]] MeshDescription getDescription() const;

private:
    void setModels(std::vector<Model*> in_models);
    void setMaterials(std::vector<Material*> in_materials);

    std::string name;
    std::vector<Model*> models{};
    std::vector<Material*> materials{};

    void updateModelDescriptions();

    std::vector<ModelDescription> model_descriptions;

    void updateRequiredMaterials();

    std::vector<std::string> required_materials;
};
