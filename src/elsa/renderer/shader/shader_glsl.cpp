/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : shader_glsl.cpp
* author      : Garra
* time        : 2019-10-01 16:03:22
* description : 
*
============================================*/


#include <vector>
#include "shader_glsl.h"
#include "glad/gl.h"
#include "../../core.h"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/string_cast.hpp"

GLSLProgram::GLSLProgram(const std::string& srcFile)
    : Shader(srcFile)
{

}

GLSLProgram::GLSLProgram(const std::string& srcVertex, const std::string& srcFragment)
    : Shader(srcVertex, srcFragment)
{
    _compile(srcVertex, srcFragment);
}

GLSLProgram::~GLSLProgram()
{
    glDeleteProgram(m_id);
}

void GLSLProgram::Bind(unsigned int slot) const 
{
    glUseProgram(m_id);
}

void GLSLProgram::Unbind() const 
{
    glUseProgram(0);
}

void GLSLProgram::_compile(const std::string& srcVertex, const std::string& srcFragment)
{

    // Create an empty vertex shader handle
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);

    // Send the vertex shader source code to GL
    // Note that std::string's .c_str is NULL character terminated.
    const GLchar *source = (const GLchar *)srcVertex.c_str();
    glShaderSource(vertexShader, 1, &source, 0);

    // Compile the vertex shader
    glCompileShader(vertexShader);

    GLint isCompiled = 0;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &isCompiled);
    if(isCompiled == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        std::vector<GLchar> infoLog(maxLength);
        glGetShaderInfoLog(vertexShader, maxLength, &maxLength, &infoLog[0]);
        
        // We don't need the shader anymore.
        glDeleteShader(vertexShader);

        // Use the infoLog as you see fit.
        
        // In this simple m_id, we'll just leave
        CORE_ERROR(infoLog.data());
        CORE_ASSERT(false, "Compile VertexShader Failed!");
        return;
    }

    CORE_TRACE("Compile VertexShader OK!");
    // Create an empty fragment shader handle
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    // Send the fragment shader source code to GL
    // Note that std::string's .c_str is NULL character terminated.
    source = (const GLchar *)srcFragment.c_str();
    glShaderSource(fragmentShader, 1, &source, 0);

    // Compile the fragment shader
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &isCompiled);
    if (isCompiled == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        std::vector<GLchar> infoLog(maxLength);
        glGetShaderInfoLog(fragmentShader, maxLength, &maxLength, &infoLog[0]);
        
        // We don't need the shader anymore.
        glDeleteShader(fragmentShader);
        // Either of them. Don't leak shaders.
        glDeleteShader(vertexShader);

        // Use the infoLog as you see fit.
        
        // In this simple m_id, we'll just leave
        CORE_ERROR(infoLog.data());
        CORE_ASSERT(false, "Compile FragmentShader Failed!");
        return;
    }
    CORE_TRACE("Compile FragmentShader OK!");

    // Vertex and fragment shaders are successfully compiled.
    // Now time to link them together into a m_id.
    // Get a m_id object.
    m_id = glCreateProgram();

    // Attach our shaders to our m_id
    glAttachShader(m_id, vertexShader);
    glAttachShader(m_id, fragmentShader);

    // Link our m_id
    glLinkProgram(m_id);

    // Note the different functions here: glGetProgram* instead of glGetShader*.
    GLint isLinked = 0;
    glGetProgramiv(m_id, GL_LINK_STATUS, (int *)&isLinked);
    if (isLinked == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetProgramiv(m_id, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        std::vector<GLchar> infoLog(maxLength);
        glGetProgramInfoLog(m_id, maxLength, &maxLength, &infoLog[0]);
        
        // We don't need the m_id anymore.
        glDeleteProgram(m_id);
        // Don't leak shaders either.
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // Use the infoLog as you see fit.
        
        // In this simple m_id, we'll just leave
        CORE_ERROR(infoLog.data());
        CORE_ASSERT(false, "LinkProgram Failed!");
        return;
    }
    CORE_TRACE("LinkProgram ({}) OK!", m_id);

    // Always detach shaders after a successful link.
    glDetachShader(m_id, vertexShader);
    glDetachShader(m_id, fragmentShader);
}

void GLSLProgram::_Upload(const char* name, const glm::mat4& matrix)
{
    Bind();
    int location = GetLocation(name);
    if(location != -1)
    {
        glad_glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
    }
}

void GLSLProgram::SetViewProjectionMatrix(const glm::mat4& vp)
{
    _Upload("u_ViewProjection", vp);
}

void GLSLProgram::SetTransform(const glm::mat4& transform)
{
    _Upload("u_Transform", transform);
}

int GLSLProgram::_UpdateLocations(const std::string& name)
{
    int location = glad_glGetAttribLocation(m_id, name.c_str());
    if(location != -1)
    {
        m_locations[name] = location;
        return location;
    }

    location = glad_glGetUniformLocation(m_id, name.c_str());
    if(location != -1)
    {
        m_locations[name] = location;
    }
    return location;
}
