#pragma once

struct OBJModelInfo
{
    std::string name;
    std::string required_material;
    std::vector<Vertex> vertices{};
    std::vector<uint32_t> indices{};
};