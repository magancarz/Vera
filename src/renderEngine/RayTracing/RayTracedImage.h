#pragma once

#include <string>

#include "RayTracerConfig.h"
#include "Models/RawModel.h"
#include "Utils/DeviceMemoryPointer.h"

class RayTracedImage
{
public:
    RayTracedImage() = default;
    RayTracedImage(const RayTracerConfig& config);
    RayTracedImage(const RayTracedImage& other);
    ~RayTracedImage();

    void clearImage();
    void saveImageToFile(const std::string& path);
    void updateImage();

    std::string name;
    int width, height;
    int max_ray_bounces;
    int generated_samples = 0;

    utils::Texture texture{};
    utils::VBO buffer{};
    cudaGraphicsResource* cuda_texture_resource;
    unsigned char* texture_data_ptr;
    dmm::DeviceMemoryPointer<unsigned long> accumulated_texture_data;

    static constexpr int COLOR_CHANNELS = 3;

private:
    void createTexture(int width, int height);

    size_t num_bytes;
};
