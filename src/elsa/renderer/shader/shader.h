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
#include <unordered_map>
#include "glm/glm.hpp"
#include "../rendererobject.h"



class Shader : public RenderObject, public std::enable_shared_from_this<Shader>
{
public:
    enum Type
    {
        UNKNOWN = -1, 
        VERTEX, 
        FRAGMENT, 
        GEOMETRY, 
        COMPUTE, 
        TESSCONTROL, 
        TESSEVALUATION, 
    };

public:
    Shader(const std::string& name) : RenderObject(name) {}
    virtual void SetViewProjectionMatrix(const glm::mat4& vp) = 0;
    virtual void SetTransformMatrix(const glm::mat4& trans) = 0;

    virtual std::shared_ptr<Shader> LoadFromFile(const std::string& srcFile) = 0;
    virtual std::shared_ptr<Shader> LoadFromSource(const std::string& srcVertex, const std::string& srcFragment) = 0;

    int GetLocation(const std::string& name);

    static std::shared_ptr<Shader> Create(const std::string& name);

protected:
    virtual int _UpdateLocations(const std::string& name) = 0;
    virtual void _Compile(const std::unordered_map<Type, std::string>& splitShaderSources) = 0;

    std::string _ReadFile(const std::string& srcFile) const ;
    std::unordered_map<Type, std::string> _SplitShaders(const std::string& sources) const;
    Type _TypeFromString(const std::string& type) const;

protected:
    std::string m_srcFile;
    std::map<const std::string, int> m_locations;
};
