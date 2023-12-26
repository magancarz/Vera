#include "RayTracedImage.h"

#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <iostream>

#include "GUI/Display.h"
#include "helper_cuda.h"

RayTracedImage::RayTracedImage(const RayTracerConfig& config)
    : name(config.image_name),
      image_config(config),
      accumulated_texture_data(image_config.image_width * image_config.image_height * COLOR_CHANNELS)
{
    if (this->image_config.image_width / this->image_config.image_height != Display::WINDOW_WIDTH / Display::WINDOW_HEIGHT)
    {
        this->image_config.image_width = 320;
        this->image_config.image_height = 200;
    }
    createTexture(image_config.image_width, image_config.image_height);
}

RayTracedImage::RayTracedImage(const RayTracedImage& other)
    : name(other.name),
      image_config(other.image_config),
      accumulated_texture_data(image_config.image_width * image_config.image_height * COLOR_CHANNELS)
{
    if (this->image_config.image_width / this->image_config.image_height != Display::WINDOW_WIDTH / Display::WINDOW_HEIGHT)
    {
        this->image_config.image_width = 320;
        this->image_config.image_height = 200;
    }
    createTexture(image_config.image_width, image_config.image_height);
}

RayTracedImage::~RayTracedImage()
{
    cudaGraphicsUnmapResources(1, &cuda_texture_resource, nullptr);
    checkCudaErrors(cudaGetLastError());
    cudaGraphicsUnregisterResource(cuda_texture_resource);
    checkCudaErrors(cudaGetLastError());
}

void RayTracedImage::clearImage()
{
    accumulated_texture_data = dmm::DeviceMemoryPointer<unsigned long>(image_config.image_width * image_config.image_height * COLOR_CHANNELS);
    generated_samples = 0;
}

void RayTracedImage::saveImageToFile(const std::string& path)
{
    const std::string temp_path = path + ".png";
    std::vector<unsigned char> texture_data(image_config.image_width * image_config.image_height * COLOR_CHANNELS);
    cudaMemcpy(texture_data.data(), texture_data_ptr, image_config.image_width * image_config.image_height * COLOR_CHANNELS, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
    stbi_write_png(
        temp_path.c_str(),
        image_config.image_width,
        image_config.image_height,
        COLOR_CHANNELS,
        texture_data.data(),
        image_config.image_width * COLOR_CHANNELS);
}

void RayTracedImage::updateImage()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer.vbo_id);
    glBindTexture(GL_TEXTURE_2D, texture.texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_config.image_width, image_config.image_height, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glGenerateMipmap(GL_TEXTURE_2D);
}

void RayTracedImage::createTexture(int width, int height)
{
    glBindTexture(GL_TEXTURE_2D, texture.texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glGenerateMipmap(GL_TEXTURE_2D);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer.vbo_id);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * COLOR_CHANNELS, nullptr, GL_DYNAMIC_COPY);
    cudaGraphicsGLRegisterBuffer(&cuda_texture_resource, buffer.vbo_id, cudaGraphicsMapFlagsWriteDiscard);
    checkCudaErrors(cudaGetLastError());

    cudaGraphicsMapResources(1, &cuda_texture_resource, nullptr);
    checkCudaErrors(cudaGetLastError());
    cudaGraphicsResourceGetMappedPointer((void**)&texture_data_ptr, &num_bytes, cuda_texture_resource);
    checkCudaErrors(cudaGetLastError());
}
