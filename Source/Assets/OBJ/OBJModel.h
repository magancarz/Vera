#pragma once

#include "Assets/Model/Model.h"
#include "Assets/AssetManager.h"
#include "OBJModelInfo.h"
#include "RenderEngine/Memory/MemoryAllocator.h"

class OBJModel : public Model
{
public:
    OBJModel(MemoryAllocator& memory_allocator, const OBJModelInfo& obj_model_info);

private:
    void createVertexBuffer(MemoryAllocator& memory_allocator, const std::vector<Vertex>& vertices);
    void createIndexBuffer(MemoryAllocator& memory_allocator, const std::vector<uint32_t>& indices);
};
