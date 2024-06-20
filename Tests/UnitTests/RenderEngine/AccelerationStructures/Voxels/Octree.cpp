#include "gtest/gtest.h"

#include "RenderEngine/AccelerationStructures/Octree/Octree.h"
#include "Assets/AssetManager.h"

#include "Environment.h"
#include "RenderEngine/AccelerationStructures/Octree/VoxelUtils.h"

TEST(OctreeTests, shouldBuildCorrectOctree)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Model* debug_model = asset_manager.fetchModel("cube");
    auto voxels = VoxelUtils::voxelize(*debug_model);

    // when
    Octree octree{5, voxels};

    // then
    //TODO: add then expect
}