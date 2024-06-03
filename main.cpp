#include "Vera.h"

#include <iostream>
#include <Logs/BasicLogger.h>
#include <Logs/LogSystem.h>

int main()
{
    LogSystem::initialize(std::make_unique<BasicLogger>());
    Vera app{};

    try
    {
    app.run();
    }
    catch (const std::exception& e)
    {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
