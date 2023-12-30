#pragma once

#include <string>

struct RayTracedImageInfo;

struct RayTracerConfig
{
    RayTracerConfig()
        : id{next_id++} {}

    RayTracerConfig(int image_width, int image_height, int number_of_ray_bounces, bool simulate_defocus_blur = false, float aperture = 0.f, float focus_distance = 1.f)
	    : image_width(image_width), image_height(image_height), number_of_ray_bounces(number_of_ray_bounces),
        simulate_defocus_blur(simulate_defocus_blur), aperture(aperture), focus_dist(focus_distance), id{next_id++} {}

    std::string image_name{"image"};
    int image_width{320};
    int image_height{200};
    int number_of_ray_bounces{1};
    int number_of_iterations{100};

    bool simulate_defocus_blur{false};
    inline static float DEFAULT_APERTURE_VALUE{0.f};
    inline static float DEFAULT_FOCUS_DISTANCE_VALUE{1.f};
    float aperture{DEFAULT_APERTURE_VALUE};
    float focus_dist{DEFAULT_FOCUS_DISTANCE_VALUE};

    size_t id;

private:
    inline static size_t next_id = 0;
};
