#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "Objects/Components/ObjectComponent.h"

class MockObjectComponent : public ObjectComponent
{
public:
    explicit MockObjectComponent(Object* object)
        : ObjectComponent(object) {}

    MOCK_METHOD(void, update, (FrameInfo& frame_info), (override));
};
