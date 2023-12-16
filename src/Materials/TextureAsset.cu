#include "TextureAsset.h"

#include <GL/glew.h>

#include "helper_cuda.h"

TextureAsset::TextureAsset(std::shared_ptr<utils::Texture> texture)
    : texture(std::move(texture))
{
    acquireTextureData();
}

void TextureAsset::bindTexture() const
{
    glBindTexture(GL_TEXTURE_2D, texture->texture_id);
}

void TextureAsset::acquireTextureData()
{
    glBindTexture(GL_TEXTURE_2D, texture->texture_id);

    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);

    texture_data_ptr = dmm::DeviceMemoryPointer<unsigned char>(width * height * NUM_OF_CHANNELS);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data_ptr.data());
}

__host__ __device__ glm::vec3 TextureAsset::getColorAtGivenUVCoordinates(const glm::vec2& uv) const
{
    const glm::vec2 fixed_uv = fixUVCoordinates(uv);
    const unsigned int pixel_index = calculatePixelIndex(fixed_uv);

    float r = static_cast<float>(texture_data_ptr[pixel_index]) / 255.f;
    float g = static_cast<float>(texture_data_ptr[pixel_index + 1]) / 255.f;
    float b = static_cast<float>(texture_data_ptr[pixel_index + 2]) / 255.f;

    return {r, g, b};
}

__host__ __device__ float TextureAsset::getAlphaValueAtGivenUVCoordinates(const glm::vec2& uv) const
{
    const glm::vec2 fixed_uv = fixUVCoordinates(uv);
    const unsigned int pixel_index = calculatePixelIndex(fixed_uv);

    return static_cast<float>(texture_data_ptr[pixel_index + 3]) / 255.f;
}

__host__ __device__ glm::vec2 TextureAsset::fixUVCoordinates(const glm::vec2& uv) const
{
    glm::vec2 fixed_uv = uv;
    if (fixed_uv.x < 0.f)
    {
        fixed_uv.x = -fixed_uv.x - glm::floor(-fixed_uv.x);
    }
    else if (fixed_uv.x >= 1.f)
    {
        fixed_uv.x -= glm::floor(fixed_uv.x);
    }

    if (fixed_uv.y < 0.f)
    {
        fixed_uv.y = -fixed_uv.y - glm::floor(-fixed_uv.y);
    }
    else if (fixed_uv.y > 1.f)
    {
        fixed_uv.y -= glm::floor(fixed_uv.y);
    }

    return fixed_uv;
}

__host__ __device__ unsigned TextureAsset::calculatePixelIndex(const glm::vec2& uv) const
{
    const auto x = static_cast<unsigned int>(uv.x * static_cast<float>(width));
    const auto y = static_cast<unsigned int>(uv.y * static_cast<float>(height));

    return (y * width + x) * NUM_OF_CHANNELS;
}
