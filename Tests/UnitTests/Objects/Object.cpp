#include "gtest/gtest.h"
#include "Objects/Object.h"

struct ObjectTests : public ::testing::Test
{
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ObjectTests, shouldCreateNewObjectWithID)
{
    // when
    Object new_object = Object::createObject();

    // then
    EXPECT_EQ(new_object.getID(), 0);
}

TEST_F(ObjectTests, shouldAssignUniqueIDToEachObject)
{
    // given
    Object first_object = Object::createObject();
    {
        Object second_object = Object::createObject();
    }

    // when
    Object third_object = Object::createObject();

    // then
    EXPECT_EQ(first_object.getID(), 1);
    EXPECT_EQ(third_object.getID(), 3);
}
