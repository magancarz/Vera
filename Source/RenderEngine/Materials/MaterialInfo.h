#pragma once

struct MaterialInfo
{
    uint64_t diffuse_texture_offset{0};
    uint64_t normal_texture_offset{0};
    alignas(4) unsigned int brightness{0};
    alignas(4) float fuzziness{-1};
    alignas(4) float refractive_index{-1};
    alignas(4) uint32_t alignment1{0};
};