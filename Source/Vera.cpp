#include "Vera.h"

#include <chrono>

#include "Input/KeyboardMovementController.h"
#include "RenderEngine/GlobalUBO.h"
#include "RenderEngine/Materials/Material.h"
#include "RenderEngine/SceneRenderers/RayTraced/RayTracedRenderer.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

void Vera::run()
{
    VulkanHelper::loadExtensionsFunctions(device.getDevice());

    std::vector<std::shared_ptr<Material>> materials = {std::make_shared<Material>(glm::vec3{0.7f, 0.7f, 0.0f})};
    world.loadObjects(device, materials);


    renderer = std::make_unique<Renderer>(window, device, world);

    auto current_time = std::chrono::high_resolution_clock::now();
    while (!window.shouldClose())
    {
        glfwPollEvents();

        auto new_time = std::chrono::high_resolution_clock::now();
        float frame_time = std::chrono::duration<float, std::chrono::seconds::period>(new_time - current_time).count();
        current_time = new_time;

        FrameInfo frame_info{};
        frame_info.frame_time = frame_time;
        world.update(frame_info);
        renderer->render(frame_info);
    }

    vkDeviceWaitIdle(device.getDevice());
}