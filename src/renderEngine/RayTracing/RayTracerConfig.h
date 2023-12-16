#pragma once

#include <string>

struct RayTracedImageInfo;

struct RayTracerConfig
{
    RayTracerConfig()
        : id{next_id++} {}

    RayTracerConfig(int image_width, int image_height, int number_of_ray_bounces)
	    : image_width(image_width), image_height(image_height), number_of_ray_bounces(number_of_ray_bounces), id{next_id++} {}

    std::string image_name{"image"};
    int image_width{320};
    int image_height{200};
    int number_of_ray_bounces{1};
    int number_of_iterations{100};
    size_t id;
    inline static size_t next_id = 0;
};
