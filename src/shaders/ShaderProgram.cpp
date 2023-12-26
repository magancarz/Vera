#include "ShaderProgram.h"

#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <sstream>
#include <iostream>

ShaderProgram::ShaderProgram(const std::string& vertex_file, const std::string& fragment_file)
{
    loadShaders(vertex_file, fragment_file);
}

ShaderProgram::~ShaderProgram()
{
    glDetachShader(program_id, vertex_shader_id);
    glDetachShader(program_id, fragment_shader_id);
    glDeleteShader(vertex_shader_id);
    glDeleteShader(fragment_shader_id);
    glDeleteProgram(program_id);
}

void ShaderProgram::start() const
{
    glUseProgram(program_id);
}

void ShaderProgram::stop()
{
    glUseProgram(0);
}

void ShaderProgram::bindAttribute(const unsigned int attribute, const char* variable_name) const
{
    glBindAttribLocation(program_id, attribute, variable_name);
}

int ShaderProgram::getUniformLocation(const std::string& uniform_name) const
{
    return glGetUniformLocation(program_id, uniform_name.c_str());
}

void ShaderProgram::loadInt(const int location, const int value)
{
    glUniform1i(location, value);
}

void ShaderProgram::loadFloat(int location, float value)
{
    glUniform1f(location, value);
}

void ShaderProgram::loadMatrix(const int location, const glm::mat4& matrix)
{
    glUniformMatrix4fv(location, 1, GL_FALSE, value_ptr(matrix));
}

void ShaderProgram::loadVector3(const int location, const glm::vec3& vector)
{
    glUniform3f(location, vector.x, vector.y, vector.z);
}

void ShaderProgram::loadVector4(int location, const glm::vec4& vector)
{
    glUniform4f(location, vector.x, vector.y, vector.z, vector.w);
}

void ShaderProgram::loadShaders(const std::string& vertex_file, const std::string& fragment_file)
{
    vertex_shader_id = loadShader(vertex_file, GL_VERTEX_SHADER);
    fragment_shader_id = loadShader(fragment_file, GL_FRAGMENT_SHADER);
    program_id = glCreateProgram();
    glAttachShader(program_id, vertex_shader_id);
    glAttachShader(program_id, fragment_shader_id);
    glLinkProgram(program_id);
    glValidateProgram(program_id);
}

unsigned int ShaderProgram::loadShader(const std::string& file, const unsigned int type) const
{
    std::ifstream stream(file);
    std::string line;
    std::stringstream ss;

    while (getline(stream, line))
    {
        ss << line << '\n';
    }
    std::string shader = ss.str();

    const char* src = shader.c_str();
    unsigned int id = glCreateShader(type);
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);
    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        auto message = static_cast<char*>(alloca(length * sizeof(char)));
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << "Failed to compile " << file << " " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") <<
            " shader!" << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(id);
        return 0;
    }

    return id;
}
