/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : mesh_opengl.h
* author      : Garra
* time        : 2019-10-06 00:34:20
* description : 
*
============================================*/


#pragma once
#include "mesh.h"

class OpenGLMesh : public Mesh
{
public:
    OpenGLMesh(const std::string& name) : Mesh(name) {}
    virtual void Bind(const std::shared_ptr<Shader>& shader) override;
};
