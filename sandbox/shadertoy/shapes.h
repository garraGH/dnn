/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/shadertoy/shapes.h
* author      : Garra
* time        : 2019-10-13 11:05:54
* description : 
*
============================================*/


#pragma once
#include "shadertoy.h"

class Shapes : public ShaderToy
{
public:
    Shapes();

    virtual std::string GetName() const { return "Shapes"; }
    virtual void OnImGuiRender() override;
    virtual std::shared_ptr<Material> GetMaterial() const override;
    virtual std::shared_ptr<Shader> GetShader() const override;


    static std::shared_ptr<Shapes> Create();

private:
    void _PrepareResources();
};
