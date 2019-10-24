/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/layer_shadertoy.h
* author      : Garra
* time        : 2019-10-10 14:26:51
* description : 
*
============================================*/


#pragma once
#include "elsa.h"
#include "shadertoy/shadertoy.h"

class ShaderToyLayer : public Layer
{
public:
    ShaderToyLayer();

    virtual void OnEvent(Event& e) override;
    virtual void OnUpdate(float deltaTime) override;
    virtual void OnImGuiRender() override;

    static std::shared_ptr<ShaderToyLayer> Create();

protected:
    void _Register(ShaderToy::Type type);
    void _PrepareResources();
    void _CreateRadioButtonOf(const std::shared_ptr<ShaderToy>& toy);

private:
    std::shared_ptr<Viewport> m_viewport = Viewport::Create("ShaderToyViewPort");
    std::shared_ptr<Renderer::Element> m_canvas = nullptr;

    ShaderToy::Type m_toyType = ShaderToy::Type::Pattern;
    std::map<ShaderToy::Type, std::shared_ptr<ShaderToy>> m_shaderToys;
};
