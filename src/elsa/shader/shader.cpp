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


