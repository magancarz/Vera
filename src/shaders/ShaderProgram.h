#pragma once

#include <glm/glm.hpp>

#include <string>

class ShaderProgram
{
public:
    ShaderProgram(const std::string& vertex_file, const std::string& fragment_file);
    virtual ~ShaderProgram();

    void start() const;
    static void stop();

    void virtual bindAttributes() = 0;
    void virtual getAllUniformLocations() = 0;

protected:
    void bindAttribute(unsigned int attribute, const char* variable_name) const;

    int getUniformLocation(const std::string& uniform_name) const;

    static void loadInt(int location, int value);
    static void loadMatrix(int location, const glm::mat4& matrix);
    static void loadVector3(int location, const glm::vec3& vector);

    void loadShaders(const std::string& vertex_file, const std::string& fragment_file);

private:
    unsigned int program_id;
    unsigned int vertex_shader_id;
    unsigned int fragment_shader_id;

    unsigned int loadShader(const std::string& file, unsigned int type) const;
};
