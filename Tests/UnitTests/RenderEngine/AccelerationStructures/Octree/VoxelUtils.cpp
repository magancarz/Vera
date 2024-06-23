#include "gtest/gtest.h"

#include "RenderEngine/AccelerationStructures/Octree/VoxelUtils.h"
#include "Assets/AssetManager.h"

#include "Environment.h"
#include "RenderEngine/AccelerationStructures/BVH/BVHTree.h"

TEST(VoxelUtilsTests, shouldConvertModelToVoxels)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Mesh* debug_mesh = asset_manager.fetchMesh("monkey");

    // when
    auto voxels = VoxelUtils::voxelize(*debug_mesh);

    // then
    //TODO: add expect
}

TEST(VoxelUtilsTests, shouldConvertBVHTreeToVoxels)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Model* debug_model = asset_manager.fetchModel("monkey");
    BVHTree bvh_tree{debug_model, 16};

    // when
    // auto voxels = VoxelUtils::voxelize(kd_tree);

    // then
    //TODO: add expect
}