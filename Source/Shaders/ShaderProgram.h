#pragma once

#include <glm/glm.hpp>

#include <string>

class ShaderProgram
{
public:
    ShaderProgram(const std::string& vertex_file, const std::string& fragment_file);
    ShaderProgram(const std::string& vertex_file, const std::string& geometry_file, const std::string& fragment_file);
    virtual ~ShaderProgram();

    void start() const;
    static void stop();

    void bindUniformBlockToShader(const std::string& uniform_block_name, unsigned int block_index) const;

    virtual void getAllUniformLocations() = 0;

protected:
    int getUniformLocation(const std::string& uniform_name) const;

    void loadInt(int location, int value) const;
    void loadFloat(int location, float value) const;
    void loadMatrix(int location, const glm::mat4& matrix) const;
    void loadVector3(int location, const glm::vec3& vector) const;

    void loadShaders(const std::string& vertex_file, const std::string& fragment_file);
    void loadShaders(const std::string& vertex_file, const std::string& geometry_file, const std::string& fragment_file);

private:
    unsigned int loadShader(const std::string& file, unsigned int type) const;
    void activateProgramIfNotActivatedYet() const;
    bool checkIfProgramIsActivated() const;

    int program_id;
    int vertex_shader_id;
    int geometry_shader_id;
    int fragment_shader_id;
};
