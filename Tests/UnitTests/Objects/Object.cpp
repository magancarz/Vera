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
    Object new_object{};

    // then
    EXPECT_EQ(new_object.getID(), 0);
}

TEST_F(ObjectTests, shouldAssignUniqueIDToEachObject)
{
    // given
    Object first_object{};
    uint32_t second_object_id{0};
    {
        Object second_object{};
        second_object_id = second_object.getID();
    }

    // when
    Object third_object{};

    // then
    EXPECT_TRUE(first_object.getID() != second_object_id);
    EXPECT_TRUE(first_object.getID() != third_object.getID());
    EXPECT_TRUE(second_object_id != third_object.getID());
}
