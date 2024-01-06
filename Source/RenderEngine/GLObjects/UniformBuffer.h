#pragma once

#include "GL/glew.h"

#include "Models/RawModel.h"

template <typename T>
class UniformBuffer
{
public:
    UniformBuffer(std::string name)
        : name(std::move(name))
    {
        prepareBuffer();
    }

    UniformBuffer(std::string name, const T& value)
        : name(std::move(name))
    {
        prepareBuffer();
        setValue(value);
    }

    std::string getName() const
    {
        return name;
    }

    void setValue(const T& val) const
    {
        glBindBuffer(GL_UNIFORM_BUFFER, buffer.vbo_id);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(T), &val, GL_STATIC_DRAW);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }

    unsigned int getUniformBlockIndex() const
    {
        return uniform_block_index;
    }

private:
    void prepareBuffer()
    {
        uniform_block_index = buffer.vbo_id;
        glBindBuffer(GL_UNIFORM_BUFFER, buffer.vbo_id);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(T), nullptr, GL_STATIC_DRAW);
        glBindBufferRange(GL_UNIFORM_BUFFER, uniform_block_index, buffer.vbo_id, 0, sizeof(T));
    }

    inline static unsigned int next_uniform_buffer_id = 0;

    std::string name;
    utils::VBO buffer;
    unsigned int uniform_block_index{0};
};