#pragma once

class MockWorld : public World
{
public:
    int getNumberOfRegisteredComponents()
    {
        uint32_t count = 0;
        for (auto& [_, registered_components] : registered_components)
        {
            count += registered_components.size();
        }

        return count;
    }
};