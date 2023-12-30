#include "ShadowMapRenderer.h"

#include <GL/glew.h>

#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"

#include "GUI/Display.h"
#include "Objects/TriangleMesh.h"

ShadowMapRenderer::ShadowMapRenderer()
{
    shadow_map_shader.getAllUniformLocations();

    createDepthMapTexture();
    createShadowMapFrameBuffer();
}

void ShadowMapRenderer::createDepthMapTexture()
{
    glGenTextures(1, &depth_map_texture);
    glBindTexture(GL_TEXTURE_2D, depth_map_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_map_width, shadow_map_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color);
}

void ShadowMapRenderer::createShadowMapFrameBuffer()
{
    glGenFramebuffers(1, &depth_map_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, depth_map_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_map_texture, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ShadowMapRenderer::renderSceneToDepthBuffer(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map)
{
    glViewport(0, 0, shadow_map_width, shadow_map_height);

    glBindFramebuffer(GL_FRAMEBUFFER, depth_map_FBO);
    glClear(GL_DEPTH_BUFFER_BIT);

    configureShaderAndMatrices();
    draw(entity_map);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT);
}

void ShadowMapRenderer::configureShaderAndMatrices()
{
    light_projection = glm::ortho(
            -shadow_map_projection_x_span, shadow_map_projection_x_span,
            -shadow_map_projection_y_span, shadow_map_projection_y_span,
            near_plane, far_plane);

    light_view = glm::lookAt(glm::vec3{0, 14, 0}, {0, 0, 0}, {0, 0, 1});

    to_light_space_transform = light_projection * light_view;
}

void ShadowMapRenderer::draw(const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map)
{
    shadow_map_shader.start();
    shadow_map_shader.loadLightSpaceMatrix(to_light_space_transform);

    for (const auto& [raw_model, entities] : entity_map)
    {
        prepareTexturedModel(raw_model);

        for (const auto& entity : entities)
        {
            prepareInstance(entity);
            glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);
        }

        unbindTexturedModel();
    }

    ShadowMapShader::stop();
}

void ShadowMapRenderer::prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const
{
    glBindVertexArray(raw_model->vao->vao_id);
    glEnableVertexAttribArray(0);
}

void ShadowMapRenderer::unbindTexturedModel()
{
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
}

void ShadowMapRenderer::prepareInstance(const std::weak_ptr<TriangleMesh>& entity) const
{
    const auto entity_rotation = entity.lock()->getRotation();
    const auto transformation_matrix = Algorithms::createTransformationMatrix
    (
            entity.lock()->getPosition(),
            entity_rotation.x,
            entity_rotation.y,
            entity_rotation.z,
            entity.lock()->getScale()
    );

    shadow_map_shader.loadTransformationMatrix(transformation_matrix);
}

void ShadowMapRenderer::bindShadowMapTexture() const
{
    glBindTexture(GL_TEXTURE_2D, depth_map_texture);
}

glm::mat4 ShadowMapRenderer::getToLightSpaceTransform() const
{
    return to_light_space_transform;
}