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
#include <memory>
#include <utility>
#include <map>
#include "../rendererobject.h"
#include "glm/glm.hpp"


class Shader : public RenderObject
{
public:
    Shader(const std::string& srcFile);
    Shader(const std::string& srcVertex, const std::string& srcFragment);

    virtual void SetViewProjectionMatrix(const glm::mat4& vp) = 0;
    virtual void SetTransform(const glm::mat4& transform) = 0;

    static std::shared_ptr<Shader> Create(const std::string& srcFile);
    static std::shared_ptr<Shader> Create(const std::string& srcVertex, const std::string& srcFragment);

    int GetLocation(const std::string& name);

protected:
    std::pair<std::string, std::string> _parseSrc(const std::string& srcFile);
    virtual void _compile(const std::string& srcVertex, const std::string& srcFragment) = 0;
    virtual int _UpdateLocations(const std::string& name) = 0;

protected:
    const std::string& m_srcFile;
    std::map<const std::string, int> m_locations;
};
