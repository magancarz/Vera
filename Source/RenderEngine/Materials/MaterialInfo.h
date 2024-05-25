#pragma once

struct MaterialInfo
{
    uint64_t texture_offset{0};
    alignas(4) unsigned int brightness{0};
    alignas(4) float fuzziness{-1};
    alignas(4) float refractive_index{-1};
    alignas(4) uint32_t alignment1{0};
};