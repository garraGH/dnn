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
    GLSLProgram(const std::string& name) : Shader(name) {}
    ~GLSLProgram(); 

    virtual void Bind(unsigned int slot=0) const override;
    virtual void Unbind() const override;

    virtual void SetViewProjectionMatrix(const glm::mat4& vp) override;
    virtual void SetTransformMatrix(const glm::mat4& trans) override;

    virtual std::shared_ptr<Shader> LoadFile(const std::string& filepath) override;
    virtual std::shared_ptr<Shader> LoadSource(const std::string& srcVertex, const std::string& srcFragment) override;

protected:
    virtual int _UpdateLocations(const std::string& name) override;
    virtual void _Compile(const std::unordered_map<Type, std::string>& splitShaderSources) override;
    
private:
    void _Upload(const char* name, const glm::mat4& matrix);
    unsigned int _ToOpenGLShaderType(Type type) const;
};
