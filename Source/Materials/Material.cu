#include "Material.h"

#include "Models/AssetManager.h"
#include "RenderEngine/RayTracing/PDF/CosinePDF.h"
#include "Utils/CurandUtils.h"
#include "renderEngine/RayTracing/Shapes/HitRecord.h"
#include "renderEngine/RayTracing/ScatterRecord.h"

Material::Material(std::string name, std::shared_ptr<TextureAsset> texture, const MaterialParameters& material_parameters,
        std::shared_ptr<TextureAsset> normal_map_texture, std::shared_ptr<TextureAsset> specular_map_texture,
        std::shared_ptr<TextureAsset> depth_map_texture)
    : name(std::move(name)), color_texture(std::move(texture)), normal_map_texture(std::move(normal_map_texture)),
	specular_map_texture(std::move(specular_map_texture)), depth_map_texture(std::move(depth_map_texture)), brightness(material_parameters.brightness),
	fuzziness(material_parameters.fuzziness), refractive_index(material_parameters.refractive_index), height_scale(material_parameters.height_scale)
{
    cuda_color_texture.copyFrom(this->color_texture.get());

    if (this->normal_map_texture)
    {
        cuda_normal_map_texture.copyFrom(this->normal_map_texture.get());
        has_normal_map = true;
    }

    if (this->specular_map_texture)
    {
        cuda_specular_map_texture.copyFrom(this->specular_map_texture.get());
        has_specular_map = true;
    }

    if (this->depth_map_texture)
    {
        cuda_depth_map_texture.copyFrom(this->depth_map_texture.get());
        has_depth_map = true;
    }
}

__device__ bool Material::scatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record) const
{
    scatter_record->color = getColor(rec->uv);
    if (has_specular_map)
    {
        const glm::vec3 specular_map_value = cuda_specular_map_texture->getColorAtGivenUVCoordinates(rec->uv);
        if (specular_map_value.r > 0.5f)
        {
            return metalScatter(r_in, rec, scatter_record);
        }

        if (specular_map_value.g > 0.5f)
        {
            return false;
        }

        if (specular_map_value.b > 0.5f)
        {
            return dielectricScatter(r_in, rec, scatter_record);
        }
    }

    return lambertianScatter(r_in, rec, scatter_record);
}

__device__ bool Material::lambertianScatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record) const
{
	scatter_record->is_specular = false;
    scatter_record->pdf = CosinePDF(r_in->curand_state, rec->normal);

    return true;
}

__device__ bool Material::metalScatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record) const
{
    const glm::vec3 reflected = reflect(normalize(r_in->direction), glm::normalize(rec->normal));
    scatter_record->specular_ray = Ray{ rec->hit_point, glm::normalize(reflected + fuzziness * randomInUnitHemisphere(r_in->curand_state, rec->normal)) };
    scatter_record->is_specular = true;

    return true;
}

__device__ float Material::scatteringPDF(const HitRecord* rec, const Ray* scattered) const
{
    const auto cosine = dot(rec->normal, normalize(scattered->direction));
    return cosine < 0.0f ? 0.0f : cosine / static_cast<float>(M_PI);
}

__device__ glm::vec3 Material::emitted(const glm::vec2& uv)
{
    return cuda_color_texture->getColorAtGivenUVCoordinates(uv) * brightness;
}

__device__ bool Material::dielectricScatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record) const
{
    scatter_record->is_specular = true;
    const float refraction_ratio = rec->front_face ? 1.0f / refractive_index : refractive_index;

    const glm::vec3 unit_direction = normalize(r_in->direction);
    const float cos_theta = dot(-unit_direction, rec->normal);
    const float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    const bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    glm::vec3 direction;

    if (cannot_refract || schlick(cos_theta, refraction_ratio) > randomFloat(r_in->curand_state))
    {
        direction = reflect(unit_direction, rec->normal);
    }
    else
    {
        refract(unit_direction, rec->normal, refraction_ratio, direction);
    }

    scatter_record->specular_ray = Ray{ rec->hit_point, direction };
    return true;
}

__device__ glm::vec3 Material::reflect(const glm::vec3& direction, const glm::vec3& normal) const
{
    return direction - 2.f * dot(direction, normal) * normal;
}

__device__ float Material::schlick(float cosine, float refractive_index) const
{
    const float r0 = (1.0f - refractive_index) / (1.0f + refractive_index);
    const float r0_squared = r0 * r0;
    return r0_squared + (1.0f - r0_squared) * pow((1.0f - cosine), 5.0f);
}

__device__ bool Material::refract(const glm::vec3& direction, const glm::vec3& normal, float refraction_ratio, glm::vec3& refracted) const
{
    const float dt = dot(direction, normal);
    const float discriminant = 1.0f - refraction_ratio * refraction_ratio * (1.0f - dt * dt);
    if (discriminant > 0.f)
    {
        refracted = refraction_ratio * (direction - normal * dt) - normal * sqrt(discriminant);
        return true;
    }
    return false;
}

__host__ __device__ glm::vec3 Material::getColor(const glm::vec2& uv) const
{
    return cuda_color_texture->getColorAtGivenUVCoordinates(uv);
}

__host__ __device__ glm::vec3 Material::getColor() const
{
    return cuda_color_texture->getColorAtGivenUVCoordinates({0.5f, 0.5f});
}

__host__ __device__ glm::vec3 Material::getLightColor() const
{
    return cuda_color_texture->getColorAtGivenUVCoordinates({0.5f, 0.5f}) * brightness;
}

__host__ __device__ float Material::getColorAlphaValue(const glm::vec2& uv) const
{
    return cuda_color_texture->getAlphaValueAtGivenUVCoordinates(uv);
}

__host__ __device__ glm::vec3 Material::getNormal(const glm::vec2& uv) const
{
    if (has_normal_map)
    {
        return cuda_normal_map_texture->getColorAtGivenUVCoordinates(uv) * 2.0f - 1.0f;
    }

    return {0.f, 1.f, 0.f};
}

__host__ __device__ glm::vec3 Material::getSpecularValue(const glm::vec2& uv) const
{
    if (has_specular_map)
    {
        return cuda_specular_map_texture->getColorAtGivenUVCoordinates(uv);
    }

    return {0.f, 0.f, 0.f};
}

__host__ __device__ float Material::getFuzziness() const
{
    return fuzziness;
}

__host__ __device__ float Material::getDepthMapHeightScale() const
{
    return height_scale;
}

void Material::bindColorTexture() const
{
    color_texture->bindTexture();
}

void Material::bindNormalMap() const
{
    if (has_normal_map)
    {
        normal_map_texture->bindTexture();
    }
}

void Material::bindDepthMap() const
{
    if (has_depth_map)
    {
        depth_map_texture->bindTexture();
    }
}