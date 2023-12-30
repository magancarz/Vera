#pragma once

#include "MaterialParameters.h"
#include "RenderEngine/RayTracing/Ray.h"
#include "TextureAsset.h"
#include "Utils/DeviceMemoryPointer.h"

struct ScatterRecord;

struct HitRecord;

class Material
{
public:
    Material(std::string name, std::shared_ptr<TextureAsset> texture, const MaterialParameters& material_parameters = {},
        std::shared_ptr<TextureAsset> normal_map_texture = nullptr, std::shared_ptr<TextureAsset> specular_map_texture = nullptr,
        std::shared_ptr<TextureAsset> depth_map_texture = nullptr);

    __device__ bool scatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record) const;
    __device__ float scatteringPDF(const HitRecord* rec, const Ray* scattered) const;
    __device__ glm::vec3 emitted(const glm::vec2& uv);
    __host__ __device__ glm::vec3 getColor(const glm::vec2& uv) const;
    __host__ __device__ float getColorAlphaValue(const glm::vec2& uv) const;
    __host__ __device__ glm::vec3 getNormal(const glm::vec2& uv) const;
    __host__ __device__ glm::vec3 getSpecularValue(const glm::vec2& uv) const;
    __host__ __device__ float getFuzziness() const;
    __host__ __device__ float getDepthMapHeightScale() const;

    __host__ __device__ bool hasNormalMap() const { return has_normal_map; }
    __host__ __device__ bool hasDepthMap() const { return has_depth_map; }

    void bindColorTexture() const;
    void bindNormalMap() const;
    void bindDepthMap() const;

    bool isEmittingLight() const { return brightness > 0.f; }

    std::string name;

private:
    __device__ bool dielectricScatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record) const;
    __device__ bool metalScatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record) const;
    __device__ bool lambertianScatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record) const;

    __device__ glm::vec3 reflect(const glm::vec3& direction, const glm::vec3& normal) const;
    __device__ float schlick(float cosine, float refractive_index) const;
    __device__ bool refract(const glm::vec3& direction, const glm::vec3& normal, float refraction_ratio, glm::vec3& refracted) const;

    std::shared_ptr<TextureAsset> color_texture;
    dmm::DeviceMemoryPointer<TextureAsset> cuda_color_texture;

    std::shared_ptr<TextureAsset> normal_map_texture;
    dmm::DeviceMemoryPointer<TextureAsset> cuda_normal_map_texture;
    bool has_normal_map{ false };

    std::shared_ptr<TextureAsset> depth_map_texture;
    dmm::DeviceMemoryPointer<TextureAsset> cuda_depth_map_texture;
    bool has_depth_map{ false };

	std::shared_ptr<TextureAsset> specular_map_texture;
    dmm::DeviceMemoryPointer<TextureAsset> cuda_specular_map_texture;
    bool has_specular_map{ false };

    float brightness{0.f};
    float fuzziness{1.f};
    float refractive_index{1.f};
    float height_scale{1.f};
};
