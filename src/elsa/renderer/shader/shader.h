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
#include "../rendererobject.h"
#include "glm/glm.hpp"

class Shader : public RenderObject
{
public:
    Shader(const std::string& srcFile);
    Shader(const std::string& srcVertex, const std::string& srcFragment);

    virtual void SetViewProjectionMatrix(const glm::mat4& vp) = 0;
    virtual void SetTransform(const glm::mat4& transform) = 0;

    static Shader* Create(const std::string& srcFile);
    static Shader* Create(const std::string& srcVertex, const std::string& srcFragment);

protected:
    std::pair<std::string, std::string> _parseSrc(const std::string& srcFile);
    virtual void _compile(const std::string& srcVertex, const std::string& srcFragment) = 0;

protected:
    const std::string& m_srcFile;
};
