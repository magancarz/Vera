#include "gtest/gtest.h"
#include "Objects/Components/TransformComponent.h"
#include "TestUtils.h"

#include "World/World.h"

struct TransformComponentTests : public ::testing::Test
{
    Object owner;
    World world;

    void SetUp() override
    {
        owner = Object{};
        world = World{};
    }

    void TearDown() override {}
};

TEST_F(TransformComponentTests, shouldReturnIdentityTransformMatrixWhenUnchanged)
{
    // given
    TransformComponent transform_component{&owner, &world};

    // when
    glm::mat4 transform = transform_component.transform();

    // then
    EXPECT_EQ(transform, glm::mat4{1.f});
}

TEST_F(TransformComponentTests, shouldReturnTransformWithCorrectRotation)
{
    // given
    TransformComponent transform_component{&owner, &world};
    transform_component.rotation.x = glm::radians(90.f);

    glm::mat4 expected_transform{1.f};
    expected_transform = glm::rotate(expected_transform, glm::radians(90.f), glm::vec3{1, 0, 0});

    // when
    glm::mat4 transform = transform_component.transform();

    // then
    TestUtils::expectTwoMatricesToBeEqual(transform, expected_transform);
}

TEST_F(TransformComponentTests, shouldReturnTransformWithCorrectTranslation)
{
    // given
    TransformComponent transform_component{&owner, &world};
    transform_component.translation.y = 10.f;

    glm::mat4 expected_transform{1.f};
    expected_transform = glm::translate(expected_transform, glm::vec3{0, 10.f, 0});

    // when
    glm::mat4 transform = transform_component.transform();

    // then
    TestUtils::expectTwoMatricesToBeEqual(transform, expected_transform);
}

TEST_F(TransformComponentTests, shouldReturnTransformWithCorrectScale)
{
    // given
    TransformComponent transform_component{&owner, &world};
    transform_component.scale = glm::vec3{3.f, 1.f, 2.f};

    glm::mat4 expected_transform{1.f};
    expected_transform = glm::scale(expected_transform, glm::vec3{3.f, 1.f, 2.f});

    // when
    glm::mat4 transform = transform_component.transform();

    // then
    TestUtils::expectTwoMatricesToBeEqual(transform, expected_transform);
}

TEST_F(TransformComponentTests, shouldReturnTransformWithCorrectRotationTranslationAndScale)
{
    // given
    TransformComponent transform_component{&owner, &world};
    transform_component.rotation.x = glm::radians(90.f);
    transform_component.translation.y = 10.f;
    transform_component.scale = glm::vec3{3.f, 1.f, 2.f};

    glm::mat4 expected_transform{1.f};
    expected_transform = glm::rotate(expected_transform, glm::radians(90.f), glm::vec3{1, 0, 0});
    expected_transform = glm::translate(expected_transform, glm::vec3{0, 10.f, 0});
    expected_transform = glm::scale(expected_transform, glm::vec3{3.f, 1.f, 2.f});

    // when
    glm::mat4 transform = transform_component.transform();

    // then
    TestUtils::expectTwoMatricesToBeEqual(transform, expected_transform);
}

TEST_F(TransformComponentTests, shouldReturnCorrectNormalMatrix)
{
    // given
    TransformComponent transform_component{&owner, &world};
    transform_component.rotation.x = glm::radians(90.f);
    transform_component.translation.y = 10.f;
    transform_component.scale = glm::vec3{3.f, 1.f, 2.f};

    glm::mat4 expected_normal_matrix{1.f};
    expected_normal_matrix = glm::rotate(expected_normal_matrix, glm::radians(90.f), glm::vec3{1, 0, 0});
    expected_normal_matrix = glm::translate(expected_normal_matrix, glm::vec3{0, 10.f, 0});
    expected_normal_matrix = glm::scale(expected_normal_matrix, glm::vec3{3.f, 1.f, 2.f});
    expected_normal_matrix = glm::transpose(glm::inverse(expected_normal_matrix));

    // when
    glm::mat4 transform = transform_component.normalMatrix();

    // then
    TestUtils::expectTwoMatricesToBeEqual(transform, expected_normal_matrix);
}