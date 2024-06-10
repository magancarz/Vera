#pragma once

struct MaterialShader
{
    uint32_t closest_hit_shader_stage_index;
    std::optional<uint32_t> any_hit_shader_stage_index;
    std::optional<uint32_t> occlusion_shader_stage_index;
};