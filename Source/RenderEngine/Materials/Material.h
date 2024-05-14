#pragma once

#include <glm/glm.hpp>

#include "RenderEngine/RenderingAPI/Buffer.h"
#include "RenderEngine/Models/ObjectDescription.h"
#include "RenderEngine/Models/BlasInstance.h"

struct MaterialInfo
{
    alignas(16) glm::vec3 color{};
    alignas(4) unsigned int brightness{0};
    alignas(4) float fuzziness{-1};
    alignas(4) float refractive_index{-1};
    alignas(4) uint32_t alignment1{0};
};

class Material
{
public:
    static std::shared_ptr<Material> loadMaterialFromFile(Device& device, const std::string& material_name);

    Material(Device& device, MaterialInfo in_material_info);

    void assignMaterialHitGroup(BlasInstance& blas_instance) const;

    void getMaterialDescription(ObjectDescription& object_description);
    [[nodiscard]] bool isLightMaterial() const { return material_info.brightness > 0; }

private:
    Device& device;
    MaterialInfo material_info;

    void createMaterialBuffer();

    std::unique_ptr<Buffer> material_info_buffer;

    void assignMaterialIndex();

    uint32_t material_index{0};
};