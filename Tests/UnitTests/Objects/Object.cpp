#include "gtest/gtest.h"

#include "Objects/Object.h"
#include "World/World.h"
#include "Objects/Components/TransformComponent.h"

#include "TestUtils.h"

TEST(ObjectTests, shouldAssignUniqueIDToEachObject)
{
    // given & when
    const Object first_object;

    uint32_t second_object_id{0};
    {
        const Object second_object;
        second_object_id = second_object.getID();
    }

    const Object third_object;

    // then
    EXPECT_TRUE(first_object.getID() != second_object_id);
    EXPECT_TRUE(first_object.getID() != third_object.getID());
    EXPECT_TRUE(second_object_id != third_object.getID());
}

TEST(ObjectTests, shouldAddComponentToObject)
{
    // given
    Object object{};

    auto transform_component = std::make_unique<TransformComponent>(object);

    // when
    object.addComponent(std::move(transform_component));

    // then
    EXPECT_TRUE(object.findComponentByClass<TransformComponent>() != nullptr);
}

TEST(ObjectTests, shouldFindObjectComponent)
{
    // given
    Object object{};

    auto transform_component = std::make_unique<TransformComponent>(object);
    object.addComponent(std::move(transform_component));

    // when
    auto found_component = object.findComponentByClass<TransformComponent>();

    // then
    EXPECT_TRUE(found_component != nullptr);
}

TEST(ObjectTests, shouldReturnObjectLocationIfTransformComponentIsPresent)
{
    // given
    Object object{};

    constexpr glm::vec3 expected_location{10, 20, 5};
    auto transform_component = std::make_unique<TransformComponent>(object);
    transform_component->translation = expected_location;
    object.addRootComponent(std::move(transform_component));

    // when
    auto actual_location = object.getLocation();

    // then
    TestUtils::expectTwoVectorsToBeEqual(actual_location, expected_location);
}
