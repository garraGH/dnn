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

GLSLProgram::~GLSLProgram()
{
    glDeleteProgram(m_id);
}

void GLSLProgram::Bind(unsigned int slot) 
{
    glUseProgram(m_id);
}

void GLSLProgram::Unbind() const 
{
    glUseProgram(0);
}

std::shared_ptr<Shader> GLSLProgram::LoadFromFile(const std::string& srcFile)
{
    m_srcFile = srcFile;
    std::string sources = _ReadFile(srcFile);
    std::unordered_map<Type, std::string> splitShaderSources = _SplitShaders(sources);
    _Compile(splitShaderSources);
    return shared_from_this();
}

std::shared_ptr<Shader> GLSLProgram::LoadFromSource(const std::string& srcVertex, const std::string& srcFragment)
{
    std::unordered_map<Type, std::string> splitShaderSources;
    splitShaderSources[VERTEX] = srcVertex;
    splitShaderSources[FRAGMENT] = srcFragment;
    _Compile(splitShaderSources);
    return shared_from_this();
}

unsigned int GLSLProgram::_ToOpenGLShaderType(Type type) const
{
    switch(type)
    {
        case VERTEX:         return GL_VERTEX_SHADER;
        case FRAGMENT:       return GL_FRAGMENT_SHADER;
        case TESSCONTROL:    return GL_TESS_CONTROL_SHADER;
        case TESSEVALUATION: return GL_TESS_EVALUATION_SHADER;
        case COMPUTE:        return GL_COMPUTE_SHADER;
        case GEOMETRY:       return GL_GEOMETRY_SHADER;
        default: CORE_ASSERT(false, "UnKnown ShaderType!"); return -1;
    }

}

void GLSLProgram::_Compile(const std::unordered_map<Type, std::string>& splitShaderSources)
{
    std::vector<unsigned int> shaderIDs;
    shaderIDs.resize(splitShaderSources.size());
    int i = 0;
    for(auto& [type, src] : splitShaderSources)
    {
        GLuint shaderID = glCreateShader(_ToOpenGLShaderType(type));
        const GLchar *source = (const GLchar *)src.c_str();
        glShaderSource(shaderID, 1, &source, 0);
        glCompileShader(shaderID);
        GLint isCompiled = 0;
        glGetShaderiv(shaderID, GL_COMPILE_STATUS, &isCompiled);
        if(isCompiled == GL_FALSE)
        {
            GLint maxLength = 0;
            glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &maxLength);
            std::vector<GLchar> infoLog(maxLength);
            glGetShaderInfoLog(shaderID, maxLength, &maxLength, &infoLog[0]);
            for(unsigned int id : shaderIDs)
            {
                glDeleteShader(id);
            }
            CORE_ERROR(infoLog.data());
            CORE_ASSERT(false, "Compile Shader Failed!");
            return;
        }
        CORE_TRACE("CompileShader ({}) OK!", shaderID);
        shaderIDs[i++] = shaderID;
    }

    GLuint programID = glCreateProgram();
    for(unsigned int shaderID : shaderIDs)
    {
        glAttachShader(programID, shaderID);
    }
    glLinkProgram(programID);

    GLint isLinked = 0;
    glGetProgramiv(programID, GL_LINK_STATUS, (int *)&isLinked);
    if (isLinked == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &maxLength);
        std::vector<GLchar> infoLog(maxLength);
        glGetProgramInfoLog(programID, maxLength, &maxLength, &infoLog[0]);
        glDeleteProgram(programID);
        for(unsigned int id : shaderIDs)
        {
            glDeleteShader(id);
        }
        CORE_ERROR(infoLog.data());
        CORE_ASSERT(false, "LinkProgram Failed!");
        return;
    }
    CORE_TRACE("LinkProgram ({}) OK!", programID);

    for(auto shaderID : shaderIDs)
    {
        glDetachShader(programID, shaderID);
    }

    m_id = programID;
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

void GLSLProgram::SetTransformMatrix(const glm::mat4& trans)
{
    _Upload("u_Transform", trans);
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
