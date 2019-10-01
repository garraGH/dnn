/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : shader_glsl.h
* author      : Garra
* time        : 2019-10-01 15:42:40
* description : 
*
============================================*/


#pragma once
#include "shader.h"

class GLSLProgram : public Shader
{
public:
    GLSLProgram(const std::string& srcFile);
    GLSLProgram(const std::string& srcVertex, const std::string& srcFragment);

    ~GLSLProgram(); 

    virtual void Bind() override;
    virtual void Unbind() override;

protected:
    virtual void _create(const std::string& srcVertex, const std::string& srcFragment) override;
};
