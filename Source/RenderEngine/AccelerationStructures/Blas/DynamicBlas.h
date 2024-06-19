#pragma once

#include "Blas.h"

class DynamicBlas : public Blas
{
public:
    DynamicBlas(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        AssetManager& asset_manager,
        const Mesh& mesh);

    DynamicBlas(const DynamicBlas&) = delete;
    DynamicBlas& operator=(const DynamicBlas&) = delete;

    void createBlas() override;

protected:
    AssetManager& asset_manager;
    const Mesh& mesh;
};
