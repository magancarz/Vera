#pragma once

struct Mesh
{
    std::string name;
    std::vector<Model*> models;
    std::vector<Material*> materials;
};