#include "gtest/gtest.h"

#include "RenderEngine/AccelerationStructures/Octree/VoxelUtils.h"
#include "Assets/AssetManager.h"

#include "Environment.h"

TEST(VoxelUtilsTests, shouldConvertModelToVoxels)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Model* debug_model = asset_manager.fetchModel(Assets::DEBUG_MESH_NAME);

    // when
    auto voxels = VoxelUtils::voxelize(*debug_model);

    // then
    //TODO: make better expect
    EXPECT_EQ(voxels.size(), 1538);
}