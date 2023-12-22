#include "Vera.h"

#include "GUI/Display.h"

int Vera::launch()
{
    editor = std::make_shared<Editor>();

    run();
    close();

    return 0;
}

void Vera::run()
{
    editor->run();
}

void Vera::close()
{
    Display::closeDisplay();
}
