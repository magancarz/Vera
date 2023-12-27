#include "Vera.h"

#include "GUI/Display.h"
#include "input/Input.h"

int Vera::launch()
{
    Display::createDisplay();
    Input::initializeInput();
    RayTracer::prepareCudaDevice();
    AssetManager::initializeAssets();

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
