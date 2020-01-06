/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/learnopengl/pbr.h
* author      : Garra
* time        : 2019-12-22 11:58:34
* description : 
*
============================================*/


#pragma once
#include "elsa.h"
#include "basicrenderelement.h"
#include "learnopengl.h"

class PBR : public LearnOpenGL
{
public:
    PBR();
    ~PBR();

    virtual void OnUpdate() override;
    virtual void OnEvent(Event& e) override;
    virtual void OnImgui() override;


protected:
    void _IrradianceConvolution();
    void _PrefilterConvolution();
    void _BRDF();

    void _UpdateUniform();
    void _Render();

private:
    glm::mat4 _CaptureView(int i);

private:
    std::shared_ptr<Renderer::Element> m_eleQuad         = Renderer::Resources::Create<REQuad>("ele_Quad");
    std::shared_ptr<Renderer::Element> m_eleSkybox       = Renderer::Resources::Create<RESkybox>("ele_Skybox");
    std::shared_ptr<Renderer::Element> m_eleCubebox      = Renderer::Resources::Create<RECubebox>("ele_Cubebox");
    std::shared_ptr<Renderer::Element> m_eleCubeboxcross = Renderer::Resources::Create<RECubeboxCross>("ele_CubeboxCross");
    std::shared_ptr<Renderer::Element> m_eleSpheres      = Renderer::Resources::Create<RESpheres>("ele_Spheres")->Set(200, 100, 2.5, 1, 40, 50);

    std::shared_ptr<Viewport> m_viewportMain = Viewport::Create("vp_main")->SetRange(0, 0, 1, 1);

};
