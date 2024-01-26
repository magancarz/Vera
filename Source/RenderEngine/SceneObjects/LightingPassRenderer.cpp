#include "LightingPassRenderer.h"

#include "Models/AssetManager.h"
#include "RenderEngine/RendererDefines.h"
#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingUtils/RenderingUtils.h"

void LightingPassRenderer::render(const std::shared_ptr<Camera>& camera, const utils::Texture& ssao)
{
    glActiveTexture(GL_TEXTURE0 + RendererDefines::G_BUFFER_STARTING_INDEX + RendererDefines::NUMBER_OF_G_BUFFER_TEXTURES + 0);
    ssao.bindTexture();

    lighting_pass_shader.loadViewPosition(camera->getPosition());
    glClear(GL_COLOR_BUFFER_BIT);
    RenderingUtils::renderQuad();
}
