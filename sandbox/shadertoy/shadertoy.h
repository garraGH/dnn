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
    enum Type
    {
        Unknown = -1, 
        FlatColor, 
        ShapingFunctions, 
        Shapes, 
        Colors, 
        Matrix, 
        HUD, 
        Pattern,
        Last, 
    };

    ShaderToy() = default;
    virtual ~ShaderToy() = default;
    
    Type GetType() const { return m_type; }
    virtual std::string GetName() const { return "ShaderToy"; }

    virtual std::shared_ptr<Material> GetMaterial() const = 0;
    virtual std::shared_ptr<Shader> GetShader() const = 0;

    virtual void OnUpdate(float deltaTime){}
    virtual void OnEvent(Event& e) {}
    virtual void OnImGuiRender(){}

    static std::shared_ptr<ShaderToy> Create(Type type);

protected:
    Type m_type;
};
