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
    Material(Device& device, MaterialInfo in_material_info);

    void assignMaterialHitGroup(BlasInstance& blas_instance);

    void getMaterialDescription(ObjectDescription& object_description);
    bool isLightMaterial() const { return material_info.brightness > 0; }

private:
    Device& device;
    MaterialInfo material_info;

    void createMaterialBuffer();

    std::unique_ptr<Buffer> material_info_buffer;
};