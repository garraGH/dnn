/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : shader.cpp
* author      : Garra
* time        : 2019-10-01 16:04:24
* description : 
*
============================================*/


#include "shader.h"
#include "../renderer.h"
#include "shader_glsl.h"
#include "../../core.h"
#include <sstream>
#include <fstream>


std::shared_ptr<Shader> Shader::Create(const std::string& name)
{
    switch(Renderer::GetAPIType())
    {
        case Renderer::API::OpenGL: return std::make_shared<GLSLProgram>(name);
        default: CORE_ASSERT(false, "API is currently unsupported."); return nullptr;
    }
}

std::shared_ptr<Shader> Shader::Define(const std::string& macro)
{
    m_macros += "#define "+macro+"\n";
    return shared_from_this();
}

int Shader::GetAttributeLocation(const std::string& name)
{
    auto result = m_attributeLocations.find(name);
    int location = result != m_attributeLocations.end()? result->second : _GetAttributeLocation(name);
//     INFO("Shader::GetAttributeLocation: {}, {}", name, location);
    return location;
}

int Shader::GetUniformLocation(const std::string& name)
{
    auto result = m_uniformLocations.find(name);
    int location =  result != m_uniformLocations.end()? result->second : _GetUniformLocation(name);
//     INFO("Shader::GetUniformLocation: {}, {}", name, location);
    return location;
}

unsigned int Shader::GetUniformBlockIndex(const std::string& name)
{
    auto result = m_uniformBlockIndices.find(name);
    unsigned int index = result != m_uniformBlockIndices.end()? result->second : _GetUniformBlockIndex(name);
//     INFO("Shader::GetUniformBlockIndex: {}, {}", name, index);
    return index;
}

std::string Shader::_ReadFile(const std::string& srcFile) const
{
    std::stringstream ss;
    std::ifstream fin(srcFile, std::ios::binary);
    CORE_ASSERT(fin, "Shader::_ReadFile Failed: "+srcFile);
    ss << fin.rdbuf();
    fin.close();

    return ss.str();
}

std::string& ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

std::string& rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(str.find_last_not_of(chars)+1);
    return str;
}

std::string& trim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    return ltrim(rtrim(str, chars), chars);
}

Shader::Type Shader::_TypeFromString(const std::string& type) const
{
    if(type == "VERTEX")                        return VERTEX;
    if(type == "FRAGMENT" || type == "PIXEL")   return FRAGMENT;
    if(type == "COMPUTE")                       return COMPUTE;
    if(type == "TESSCONTROL")                   return TESSCONTROL;
    if(type == "TESSEVALUATION")                return TESSEVALUATION;
    if(type == "GEOMETRY")                      return GEOMETRY;

    CORE_ASSERT(false, "Shader::_TypeFromString: UnKnown shader type "+type);
    return UNKNOWN;
}

std::unordered_map<Shader::Type, std::string> Shader::_SplitShaders(const std::string& sources) const
{
    std::unordered_map<Type, std::string> splitShaderSources;

    const char* typeToken = "#type";
    const char* versionToken = "#version";
    size_t typeTokenLength = strlen(typeToken);
    size_t pos = sources.find(typeToken, 0);

    while(pos != std::string::npos)
    {
        size_t eol = sources.find_first_of("\r\n", pos);
        CORE_ASSERT(eol != std::string::npos, "Syntax error!");
        size_t begin = pos+typeTokenLength+1;
        std::string type = sources.substr(begin, eol-begin);
        trim(type);
        transform(type.begin(), type.end(), type.begin(), toupper);
        pos = sources.find(versionToken, eol);
        eol = sources.find_first_of("\r\n", pos);
        size_t nextLinePos = sources.find_first_not_of("\r\n", eol);
        splitShaderSources[_TypeFromString(type)] = sources.substr(pos, nextLinePos-pos)+m_macros;
        pos = sources.find(typeToken, nextLinePos);
        splitShaderSources[_TypeFromString(type)] += sources.substr(nextLinePos, pos-(nextLinePos == std::string::npos? sources.size()-1 : nextLinePos));
    }

    return splitShaderSources;
}

