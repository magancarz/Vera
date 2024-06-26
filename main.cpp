#include "Vera.h"

#include <iostream>

#include "Logs/LoggerDecorator.h"

int main()
{
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
