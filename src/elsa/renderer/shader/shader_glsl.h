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

    virtual void Bind(unsigned int slot=0) override;
    virtual void Unbind() const override;

    virtual void SetWorld2ClipMatrix(const glm::mat4& w2c) override;
    virtual void SetModel2WorldMatrix(const glm::mat4& m2w) override;

    virtual std::shared_ptr<Shader> LoadFromFile(const std::string& filepath) override;
    virtual std::shared_ptr<Shader> LoadFromSource(const std::string& srcVertex, const std::string& srcFragment) override;

protected:
    virtual int _GetAttributeLocation(const std::string& name) override;
    virtual int _GetUniformLocation(const std::string& name) override;
    virtual unsigned int _GetUniformBlockIndex(const std::string& name) override;

    virtual void _Compile(const std::unordered_map<Type, std::string>& splitShaderSources) override;
    
    std::string _GetStringType(Type type) const ;
private:
    void _Upload(const char* name, const glm::mat4& matrix);
    unsigned int _ToOpenGLShaderType(Type type) const;
};
