#include "RenderingUtils.h"

#include "Models/AssetManager.h"

void RenderingUtils::renderQuad()
{
    glBindVertexArray(AssetManager::texture_quad.vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glEnable(GL_DEPTH_TEST);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
}

void RenderingUtils::renderYInvertedQuad()
{
    glBindVertexArray(AssetManager::y_inverse_texture_quad.vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glEnable(GL_DEPTH_TEST);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
}