#include "gtest/gtest.h"

#include "RenderEngine/AccelerationStructures/KD/KDTree.h"

#include <Environment.h>

#include "Assets/AssetManager.h"
#include "RenderEngine/AccelerationStructures/BVH/BVHTree.h"

TEST(BVHTests, shouldBuildCorrectBVHTree)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Model* debug_model = asset_manager.fetchModel("dragon");

    // when
    BVHTree bvh_tree{debug_model, 16};

    // then
    //TODO: add expect
}