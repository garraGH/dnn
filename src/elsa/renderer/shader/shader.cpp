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

Shader::Shader(const std::string& srcFile)
    : m_srcFile(srcFile)
{

}

Shader::Shader(const std::string& srcVertex, const std::string& srcFragment)
    : m_srcFile("")
{

}

std::pair<std::string, std::string> Shader::_parseSrc(const std::string& srcFile)
{
    std::string srcVertex;
    std::string srcFragment;

    return { srcVertex, srcFragment };
}


std::shared_ptr<Shader> Shader::Create(const std::string& srcFile)
{
    switch(Renderer::GetAPIType())
    {
        case Renderer::API::OpenGL: return std::make_shared<GLSLProgram>(srcFile);
        default: CORE_ASSERT(false, "API is currently unsupported."); return nullptr;
    }
}


std::shared_ptr<Shader> Shader::Create(const std::string& srcVertex, const std::string& srcFragment)
{
    
    switch(Renderer::GetAPIType())
    {
        case Renderer::API::OpenGL: return std::make_shared<GLSLProgram>(srcVertex, srcFragment);
        default: CORE_ASSERT(false, "API is currently unsupported."); return nullptr;
    }
}

int Shader::GetLocation(const std::string& name)
{
    auto result = m_locations.find(name);
    int location =  result != m_locations.end()? result->second : _UpdateLocations(name);
    INFO("Shader::GetLocation: {}, {}", name, location);
    return location;
}


