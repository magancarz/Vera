#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "Models/RawModel.h"
#include "Utils/DeviceMemoryPointer.h"

namespace utils {
    class Texture;
}

class TextureAsset
{
public:
    TextureAsset(std::shared_ptr<utils::Texture> texture);

    void bindTexture() const;

    __host__ __device__ glm::vec3 getColorAtGivenUVCoordinates(const glm::vec2& uv) const;
    __host__ __device__ float getAlphaValueAtGivenUVCoordinates(const glm::vec2& uv) const;

    inline static constexpr uint8_t NUM_OF_CHANNELS = 4;

private:
    void acquireTextureData();
    static __host__ __device__ glm::vec2 fixUVCoordinates(const glm::vec2& uv);
    __host__ __device__ unsigned int calculatePixelIndex(const glm::vec2& uv) const;

    std::shared_ptr<utils::Texture> texture;
    dmm::DeviceMemoryPointer<unsigned char> texture_data_ptr;
    int width, height;
};
