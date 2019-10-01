/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : shader.h
* author      : Garra
* time        : 2019-10-01 15:42:40
* description : 
*
============================================*/


#pragma once
#include <string>
#include <utility>
class Shader
{
public:
    Shader(const std::string& srcFile);
    Shader(const std::string& srcVertex, const std::string& srcFragment);
    virtual ~Shader() {};
    

    virtual void Bind() = 0;
    virtual void Unbind() = 0;
protected:
    std::pair<std::string, std::string> _parseSrc(const std::string& srcFile);
    virtual void _create(const std::string& srcVertex, const std::string& srcFragment) = 0;

protected:
    const std::string& m_srcFile;
    unsigned int m_id = 0;
};
