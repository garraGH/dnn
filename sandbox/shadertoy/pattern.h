/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : pattern.h
* author      : Garra
* time        : 2019-10-19 12:10:29
* description : 
*
============================================*/


#pragma once

#include "shadertoy.h"

class Pattern : public ShaderToy
{
public:
    Pattern();

    virtual std::string GetName() const { return "Pattern"; }

    virtual void OnUpdate(float deltaTime) override;
    virtual void OnEvent(Event& e) override;
    virtual void OnImGuiRender() override;

    virtual std::shared_ptr<Material> GetMaterial() const override;
    virtual std::shared_ptr<Shader> GetShader() const override;

    static std::shared_ptr<Pattern> Create();
protected:
private:
    void _PrepareResources();

private:
    float* m_color = nullptr;
    float* m_tiles = nullptr;
    float* m_time = nullptr;

};
