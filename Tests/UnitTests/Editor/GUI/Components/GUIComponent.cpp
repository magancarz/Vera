#include "gtest/gtest.h"

#include <Editor/GUI/Components/GUIContainer.h>
#include "Mocks/MockGUIComponent.h"

using ::testing::_;

TEST(GUIComponentTests, shouldUpdateTreeComponents)
{
    // given
    auto first_component = std::make_unique<MockGUIComponent>("first_component");
    MockGUIComponent& first_component_ref = *first_component;
    auto second_component = std::make_unique<MockGUIComponent>("second_component");
    MockGUIComponent& second_component_ref = *second_component;
    auto third_component = std::make_unique<MockGUIComponent>("third_component");
    MockGUIComponent& third_component_ref = *third_component;

    auto sub_container = std::make_unique<GUIContainer>("sub_container");
    sub_container->addComponent(std::move(first_component));
    sub_container->addComponent(std::move(second_component));

    GUIContainer main_container{"main_container"};
    main_container.addComponent(std::move(third_component));
    main_container.addComponent(std::move(sub_container));

    FrameInfo frame_info{};

    // then
    EXPECT_CALL(first_component_ref, update(_)).Times(1);
    EXPECT_CALL(second_component_ref, update(_)).Times(1);
    EXPECT_CALL(third_component_ref, update(_)).Times(1);

    // when
    main_container.update(frame_info);
}
