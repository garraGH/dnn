/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/shadertoy/flatcolor.h
* author      : Garra
* time        : 2019-10-13 11:05:56
* description : 
*
============================================*/


#pragma once
#include "shadertoy.h"

class FlatColor : public ShaderToy
{
public:
    FlatColor();

    virtual std::string GetName() const override { return "FlatColor"; }
    virtual std::shared_ptr<Material> GetMaterial() const override;
    virtual std::shared_ptr<Shader> GetShader() const override;
    virtual void OnImGuiRender() override;

    static std::shared_ptr<FlatColor> Create();

private:
    void _PrepareResources();
};
