#include "gtest/gtest.h"

#include "RenderEngine/AccelerationStructures/Octree/Octree.h"
#include "Assets/AssetManager.h"

#include "Environment.h"
#include "RenderEngine/AccelerationStructures/Octree/VoxelUtils.h"

TEST(OctreeTests, shouldBuildCorrectOctree)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Mesh* debug_mesh = asset_manager.fetchMesh("cube");
    auto voxels = VoxelUtils::voxelize(*debug_mesh);

    // when
    Octree octree{5, voxels};

    // then
    //TODO: add then expect
}