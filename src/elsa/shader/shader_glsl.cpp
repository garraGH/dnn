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
#include "../core.h"

GLSLProgram::GLSLProgram(const std::string& srcFile)
    : Shader(srcFile)
{

}

GLSLProgram::GLSLProgram(const std::string& srcVertex, const std::string& srcFragment)
    : Shader(srcVertex, srcFragment)
{
    _create(srcVertex, srcFragment);
}

GLSLProgram::~GLSLProgram()
{
    glDeleteProgram(m_id);
}

void GLSLProgram::Bind()
{
    glUseProgram(m_id);
}

void GLSLProgram::Unbind()
{
    glUseProgram(0);
}

void GLSLProgram::_create(const std::string& srcVertex, const std::string& srcFragment)
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

