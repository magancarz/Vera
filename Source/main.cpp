#include "Vera.h"

#include <iostream>

int main()
{
    Vera app{};

    try
    {
        app.launch();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}