/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/src/elsa/renderer/material/material_opengl.h
* author      : Garra
* time        : 2019-10-05 22:48:54
* description : 
*
============================================*/


#pragma once
#include "material.h"

class OpenGLMaterial : public Material
{
public:
    OpenGLMaterial(const std::string& name) : Material(name) {}
    virtual void Bind(const std::shared_ptr<Shader>& shader) override;
    static std::shared_ptr<Material> Create(const std::string& name);

private:
    void _BindAttribute(const std::shared_ptr<Shader>& shader);
    void _BindTexture(const std::shared_ptr<Shader>& shader);
};
