#pragma once

#include <iostream>
#include <GL/glew.h>

#include <vector>

#include "TriangleData.h"

namespace utils
{
    class VAO
    {
    public:
        unsigned int vao_id{0};

        VAO()
        {
            glGenVertexArrays(1, &vao_id);
        }

        ~VAO()
        {
            glDeleteVertexArrays(1, &vao_id);
        }
    };

    class VBO
    {
    public:
        unsigned int vbo_id{0};

        VBO()
        {
            glGenBuffers(1, &vbo_id);
        }

        ~VBO()
        {
            glDeleteBuffers(1, &vbo_id);
        }
    };

    class Texture
    {
    public:
        unsigned int texture_id{0};
        GLenum type{GL_TEXTURE_2D};

        Texture()
        {
            glGenTextures(1, &texture_id);
        }

        ~Texture()
        {
            glDeleteTextures(1, &texture_id);
        }

        void setAsCubeMapTexture()
        {
            type = GL_TEXTURE_CUBE_MAP;
        }
    };

    class FBO
    {
    public:
        unsigned int FBO_id{0};

        FBO()
        {
            glGenFramebuffers(1, &FBO_id);
        }

        ~FBO()
        {
            glDeleteFramebuffers(1, &FBO_id);
        }
    };
}

struct RawModel
{
    std::string model_name;
    std::shared_ptr<utils::VAO> vao;
    unsigned int vertex_count;
    std::vector<TriangleData> triangles;

    bool operator<(const std::weak_ptr<RawModel>& other) const
    {
        return vao->vao_id > other.lock()->vao->vao_id;
    }
};
