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

void SkyboxRenderer::renderSkybox(
        const utils::FBO& hdr_fbo,
        const std::shared_ptr<Camera>& camera,
        const std::shared_ptr<utils::Texture>& skybox)
{
    hdr_fbo.bindFramebuffer();

    skybox_shader.start();
    skybox_shader.loadViewMatrix(camera);
    skybox_shader.loadProjectionMatrix(camera);

    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
    glActiveTexture(GL_TEXTURE0);
    skybox->bindTexture();
    glBindVertexArray(cube_model->vao->vao_id);
    glEnableVertexAttribArray(0);
    glDrawElements(GL_TRIANGLES, static_cast<int>(cube_model->vertex_count), GL_UNSIGNED_INT, nullptr);
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
    glEnable(GL_CULL_FACE);
    glDepthFunc(GL_LESS);

    hdr_fbo.unbind();
}

void SkyboxRenderer::renderSkybox(
        const utils::FBO& hdr_fbo,
        const std::shared_ptr<Camera>& camera)
{
    hdr_fbo.bindFramebuffer();

    skybox_shader.start();
    skybox_shader.loadViewMatrix(camera);
    skybox_shader.loadProjectionMatrix(camera);

    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
    glBindVertexArray(cube_model->vao->vao_id);
    glEnableVertexAttribArray(0);
    glDrawElements(GL_TRIANGLES, static_cast<int>(cube_model->vertex_count), GL_UNSIGNED_INT, nullptr);
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
    glEnable(GL_CULL_FACE);
    glDepthFunc(GL_LESS);

    hdr_fbo.unbind();
}