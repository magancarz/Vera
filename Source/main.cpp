#include "Vera.h"

#include <iostream>

#include "Input/Input.h"

int main()
{
    Display::createDisplay();
    Input::initializeInput();

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