/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/shadertoy/shadertoy.h
* author      : Garra
* time        : 2019-10-13 10:48:16
* description : 
*
============================================*/


#pragma once
#include <memory>
#include "elsa.h"

class ShaderToy
{
public:
    enum class Type
    {
        FlatColor, 
        ShapingFunctions, 
        Shapes, 
        Colors
    };

    ShaderToy() = default;
    virtual ~ShaderToy() = default;
    
    Type GetType() const { return m_type; }
    virtual std::string GetName() const { return "ShaderToy"; }

    virtual std::shared_ptr<Material> GetMaterial() const = 0;
    virtual std::shared_ptr<Shader> GetShader() const = 0;

    virtual void OnUpdate(){}
    virtual void OnImGuiRender(){}

    static std::shared_ptr<ShaderToy> Create(Type type);

protected:
    Type m_type;
};
