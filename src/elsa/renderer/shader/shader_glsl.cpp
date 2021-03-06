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
    splitShaderSources[Type::VERTEX] = srcVertex;
    splitShaderSources[Type::FRAGMENT] = srcFragment;
    _Compile(splitShaderSources);
    return shared_from_this();
}

unsigned int GLSLProgram::_ToOpenGLShaderType(Type type) const
{
    switch(type)
    {
        case Type::VERTEX:         return GL_VERTEX_SHADER;
        case Type::FRAGMENT:       return GL_FRAGMENT_SHADER;
        case Type::TESSCONTROL:    return GL_TESS_CONTROL_SHADER;
        case Type::TESSEVALUATION: return GL_TESS_EVALUATION_SHADER;
        case Type::COMPUTE:        return GL_COMPUTE_SHADER;
        case Type::GEOMETRY:       return GL_GEOMETRY_SHADER;
        default: CORE_ASSERT(false, "UnKnown ShaderType!"); return -1;
    }

}

std::string GLSLProgram::_GetStringType(Type type) const
{
#define CASE(x) case x: return #x;
    switch(type)
    {
        CASE(Type::VERTEX);
        CASE(Type::FRAGMENT);
        CASE(Type::TESSCONTROL);
        CASE(Type::TESSEVALUATION);
        CASE(Type::COMPUTE);
        CASE(Type::GEOMETRY);
        default: CORE_ASSERT(false, "UnKnown ShaderType!"); return "";
    }
#undef CASE
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
            CORE_ERROR("{}", infoLog.data());
            CORE_ASSERT(false, "Shader({}-{}) Compile failed!", m_srcFile, _GetStringType(type));
            return;
        }
        CORE_TRACE("Shader({}-{}-{}) Compile OK!", m_srcFile, _GetStringType(type), shaderID);
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
        CORE_ERROR("{}", infoLog.data());
        CORE_ASSERT(false, "Program({}) Link Failed!", m_srcFile);
        return;
    }
    CORE_TRACE("Program({}-{}) Link OK!", m_srcFile, programID);

    for(auto shaderID : shaderIDs)
    {
        glDetachShader(programID, shaderID);
    }

    m_id = programID;
}


void GLSLProgram::_Upload(const char* name, const glm::mat4& matrix)
{
    Bind();
    int location = GetUniformLocation(name);
    if(location != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
    }
}

void GLSLProgram::SetWorld2ClipMatrix(const glm::mat4& w2c)
{
    _Upload("u_WS2CS", w2c);
}

void GLSLProgram::SetModel2WorldMatrix(const glm::mat4& m2w)
{
    _Upload("u_MS2WS", m2w);
}

int GLSLProgram::_GetAttributeLocation(const std::string& name)
{
    int location = glGetAttribLocation(m_id, name.c_str());
    m_attributeLocations[name] = location;
    return location;
}

int GLSLProgram::_GetUniformLocation(const std::string& name)
{
    int location = glGetUniformLocation(m_id, name.c_str());
    m_uniformLocations[name] = location;
    return location;
}

unsigned int GLSLProgram::_GetUniformBlockIndex(const std::string& name)
{
    unsigned int index = glGetUniformBlockIndex(m_id, name.c_str());
    m_uniformBlockIndices[name] = index;
    return index;
}
