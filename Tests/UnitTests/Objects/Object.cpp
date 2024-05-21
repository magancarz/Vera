#include "gtest/gtest.h"

#include "Objects/Object.h"
#include "World/World.h"
#include "TestUtils.h"

struct ObjectTests : public ::testing::Test
{
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ObjectTests, shouldAssignUniqueIDToEachObject)
{
    // given & when
    Object first_object{};

    uint32_t second_object_id{0};
    {
        Object second_object{};
        second_object_id = second_object.getID();
    }

    Object third_object{};

    // then
    EXPECT_TRUE(first_object.getID() != second_object_id);
    EXPECT_TRUE(first_object.getID() != third_object.getID());
    EXPECT_TRUE(second_object_id != third_object.getID());
}

TEST_F(ObjectTests, shouldAddComponentToObject)
{
    // given
    Object object{};

    auto transform_component = std::make_shared<TransformComponent>(&object);

    // when
    object.addComponent(transform_component);

    // then
    EXPECT_TRUE(object.findComponentByClass<TransformComponent>() != nullptr);
}

TEST_F(ObjectTests, shouldFindObjectComponent)
{
    // given
    Object object{};

    auto transform_component = std::make_shared<TransformComponent>(&object);
    object.addComponent(transform_component);

    // when
    auto found_component = object.findComponentByClass<TransformComponent>();

    // then
    EXPECT_TRUE(found_component != nullptr);
}

TEST_F(ObjectTests, shouldReturnObjectLocationIfTransformComponentIsPresent)
{
    // given
    Object object{};

    const glm::vec3 expected_location{10, 20, 5};
    auto transform_component = std::make_shared<TransformComponent>(&object);
    transform_component->translation = expected_location;
    object.addComponent(transform_component);

    // when
    auto actual_location = object.getLocation();

    // then
    TestUtils::expectTwoVectorsToBeEqual(actual_location, expected_location);
}

TEST_F(ObjectTests, shouldThrowAssertIfUserWantedToObtainObjectDescriptionWhileProperComponentsAreNotPresentDeathTest)
{
    // given
    Object object{};

    // when & then
    EXPECT_DEATH(object.getObjectDescription(), ".*");
}