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
    enum class Type
    {
        UNKNOWN = -1, 
        VERTEX, 
        FRAGMENT, 
        GEOMETRY, 
        COMPUTE, 
        TESSCONTROL, 
        TESSEVALUATION, 
    };

    enum class Macro : int
    {
        UNKOWN                  = -1, 
        INSTANCE                = 0x01<<0,
        DIFFUSE_REFLECTANCE     = 0x01<<1, 
        SPECULAR_REFLECTANCE    = 0x01<<2, 
        EMISSIVE_COLOR          = 0x01<<3, 
        DIFFUSE_MAP             = 0x01<<4, 
        SPECULAR_MAP            = 0x01<<5, 
        EMISSIVE_MAP            = 0x01<<6, 
        NORMAL_MAP              = 0x01<<7,
        DISPLACEMENT_MAP        = 0x01<<8, 
        PARALLAX_MAP            = DISPLACEMENT_MAP, 
        HEIGHT_MAP              = 0x01<<9, 
        AMBIENET_OCCLUSION_MAP  = 0x01<<10,  
        REFLECTION_MAP          = 0x01<<11, 
        SHININESS_MAP           = 0x01<<12, 
        METALLIC_MAP            = 0x01<<13, 
        ROUGHNESS_MAP           = 0x01<<14, 
        AO_MAP                  = 0x01<<15, 
        ALBEDO_MAP              = 0x01<<16, 
        IRRADIANCE_DIFFUSE_MAP  = 0x01<<17, 
        IRRADIANCE_SPECULAR_MAP = 0x01<<18, 
        TONE_MAP                = 0x01<<19, 
        GAMMA_CORRECTION        = 0x01<<20,
    };

public:
    Shader(const std::string& name) : RenderObject(name) {}
    virtual void SetWorld2ClipMatrix(const glm::mat4& w2c) = 0;
    virtual void SetModel2WorldMatrix(const glm::mat4& m2w) = 0;

    virtual std::shared_ptr<Shader> LoadFromFile(const std::string& srcFile) = 0;
    virtual std::shared_ptr<Shader> LoadFromSource(const std::string& srcVertex, const std::string& srcFragment) = 0;

    static std::string GetTypeName() { return "Shader"; }

    int GetAttributeLocation(const std::string& name);
    int GetUniformLocation(const std::string& name);
    unsigned int GetUniformBlockIndex(const std::string& name);

    std::shared_ptr<Shader> Define(const std::string& macro);
    std::shared_ptr<Shader> Define(int macros);

    static std::shared_ptr<Shader> Create(const std::string& name);

protected:
    virtual int _GetUniformLocation(const std::string& name) = 0;
    virtual int _GetAttributeLocation(const std::string& name) = 0;
    virtual unsigned int _GetUniformBlockIndex(const std::string& name) = 0;
    virtual void _Compile(const std::unordered_map<Type, std::string>& splitShaderSources) = 0;

    std::string _ReadFile(const std::string& srcFile) const ;
    std::unordered_map<Type, std::string> _SplitShaders(const std::string& sources);
    Type _TypeFromString(const std::string& type) const;

protected:
    std::string m_srcFile;
    std::string m_macros;
    std::map<const std::string, int> m_attributeLocations;
    std::map<const std::string, int> m_uniformLocations;
    std::map<const std::string, unsigned int> m_uniformBlockIndices;
};
