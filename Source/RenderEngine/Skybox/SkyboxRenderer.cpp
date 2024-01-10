#include "SkyboxRenderer.h"

#include "Models/AssetManager.h"

SkyboxRenderer::SkyboxRenderer()
{
    skybox_shader.getAllUniformLocations();
    cube_model = AssetManager::findModelAsset("cube");
    cube_map = AssetManager::loadCubeMap({
        "nightRight",
        "nightLeft",
        "nightTop",
        "nightBottom",
        "nightBack",
        "nightFront",
    });
}

void SkyboxRenderer::renderSkybox(const std::shared_ptr<Camera>& camera, const std::shared_ptr<utils::Texture>& skybox)
{
    skybox_shader.start();
    skybox_shader.loadViewMatrix(camera);
    skybox_shader.loadProjectionMatrix(camera);

    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glActiveTexture(GL_TEXTURE0);
    skybox->bindTexture();
    glBindVertexArray(cube_model->vao->vao_id);
    glEnableVertexAttribArray(0);
    glDrawElements(GL_TRIANGLES, static_cast<int>(cube_model->vertex_count), GL_UNSIGNED_INT, nullptr);
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
    glEnable(GL_CULL_FACE);
    glDepthMask(GL_TRUE);
}

void SkyboxRenderer::renderSkybox(const std::shared_ptr<Camera>& camera)
{
    skybox_shader.start();
    skybox_shader.loadViewMatrix(camera);
    skybox_shader.loadProjectionMatrix(camera);

    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glBindVertexArray(cube_model->vao->vao_id);
    glEnableVertexAttribArray(0);
    glDrawElements(GL_TRIANGLES, static_cast<int>(cube_model->vertex_count), GL_UNSIGNED_INT, nullptr);
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
    glEnable(GL_CULL_FACE);
    glDepthMask(GL_TRUE);
}