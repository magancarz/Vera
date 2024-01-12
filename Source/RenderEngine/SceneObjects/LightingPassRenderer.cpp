#include "LightingPassRenderer.h"

#include "Models/AssetManager.h"
#include "RenderEngine/RendererDefines.h"

LightingPassRenderer::LightingPassRenderer()
{
    lighting_pass_shader.getAllUniformLocations();
    lighting_pass_shader.connectTextureUnits();

    quad = AssetManager::loadSimpleModel(quad_positions, quad_textures);
}

void LightingPassRenderer::render(
        const utils::Texture& g_position,
        const utils::Texture& g_normal,
        const utils::Texture& g_color_spec)
{
    glActiveTexture(GL_TEXTURE0 + RendererDefines::G_BUFFER_STARTING_INDEX + 0);
    g_position.bindTexture();
    glActiveTexture(GL_TEXTURE0 + RendererDefines::G_BUFFER_STARTING_INDEX + 1);
    g_normal.bindTexture();
    glActiveTexture(GL_TEXTURE0 + RendererDefines::G_BUFFER_STARTING_INDEX + 2);
    g_color_spec.bindTexture();

    glBindVertexArray(quad.vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glDisable(GL_DEPTH_TEST);

    lighting_pass_shader.start();
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glEnable(GL_DEPTH_TEST);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
}
