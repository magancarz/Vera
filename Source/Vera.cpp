#include "Vera.h"

#include "GUI/Display.h"
#include "input/Input.h"

int Vera::launch()
{
    Display::createDisplay();
    Input::initializeInput();

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
    //TODO:
    //vkDeviceWaitIdle(device.getDevice());
    Display::closeDisplay();
}
