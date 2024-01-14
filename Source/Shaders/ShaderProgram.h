#pragma once

#include <glm/glm.hpp>

#include <string>
#include "UniformBuffer.h"

class ShaderProgram
{
public:
    ShaderProgram(const std::string& vertex_file, const std::string& fragment_file);
    ShaderProgram(const std::string& vertex_file, const std::string& geometry_file, const std::string& fragment_file);
    virtual ~ShaderProgram();

    void start() const;
    static void stop();

    template <typename T>
    void bindUniformBuffer(const UniformBuffer<T>& uniform_buffer)
    {
        int shader_uniform_block_index = glGetUniformBlockIndex(program_id, uniform_buffer.getName().c_str());
        if (shader_uniform_block_index == -1)
        {
            return;
        }
        glUniformBlockBinding(program_id, shader_uniform_block_index, uniform_buffer.getUniformBlockIndex());
    }

    virtual void connectTextureUnits() {}
    virtual void getAllUniformLocations() {};

protected:
    int getUniformLocation(const std::string& uniform_name) const;

    void loadInt(int location, int value) const;
    void loadFloat(int location, float value) const;
    void loadMatrix(int location, const glm::mat4& matrix) const;
    void loadVector3(int location, const glm::vec3& vector) const;

    void loadShaders(const std::string& vertex_file, const std::string& fragment_file);
    void loadShaders(const std::string& vertex_file, const std::string& geometry_file, const std::string& fragment_file);

    int program_id;
    int vertex_shader_id;
    int geometry_shader_id;
    int fragment_shader_id;

private:
    unsigned int loadShader(const std::string& file, unsigned int type) const;
    void activateProgramIfNotActivatedYet() const;
    bool checkIfProgramIsActivated() const;

    inline static int last_used_shader_program{-1};
};
