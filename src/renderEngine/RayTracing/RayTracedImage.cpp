#include "RayTracedImage.h"

#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <iostream>

#include "GUI/Display.h"

RayTracedImage::RayTracedImage(const RayTracerConfig& config)
    : name(config.image_name),
      width(config.image_width),
      height(config.image_height),
      max_ray_bounces(config.number_of_ray_bounces),
      accumulated_texture_data(width * height * COLOR_CHANNELS)
{
    if (this->width / this->height != Display::WINDOW_WIDTH / Display::WINDOW_HEIGHT)
    {
        this->width = 320;
        this->height = 200;
    }
    createTexture(width, height);
}

RayTracedImage::RayTracedImage(const RayTracedImage& other)
    : name(other.name),
      width(other.width),
      height(other.height),
      max_ray_bounces(other.max_ray_bounces),
      accumulated_texture_data(width * height * COLOR_CHANNELS)
{
    if (this->width / this->height != Display::WINDOW_WIDTH / Display::WINDOW_HEIGHT)
    {
        this->width = 320;
        this->height = 200;
    }
    createTexture(width, height);
}

RayTracedImage::~RayTracedImage()
{
    cudaGraphicsUnmapResources(1, &cuda_texture_resource, nullptr);
    cudaGraphicsUnregisterResource(cuda_texture_resource);
}

void RayTracedImage::clearImage()
{
    accumulated_texture_data = dmm::DeviceMemoryPointer<unsigned long>(width * height * COLOR_CHANNELS);
    generated_samples = 0;
}

void RayTracedImage::saveImageToFile(const std::string& path)
{
    const std::string temp_path = path + ".png";
    std::vector<unsigned char> texture_data(width * height * COLOR_CHANNELS);
    cudaMemcpy(texture_data.data(), texture_data_ptr, width * height * COLOR_CHANNELS, cudaMemcpyDeviceToHost);
    stbi_write_png(
        temp_path.c_str(),
        width,
        height,
        COLOR_CHANNELS,
        texture_data.data(),
        width * COLOR_CHANNELS);
}

void RayTracedImage::updateImage()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer.vbo_id);
    glBindTexture(GL_TEXTURE_2D, texture.texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
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

    cudaGraphicsMapResources(1, &cuda_texture_resource, nullptr);
    cudaGraphicsResourceGetMappedPointer((void**)&texture_data_ptr, &num_bytes, cuda_texture_resource);
}
