/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/learnopengl/bloom.h
* author      : Garra
* time        : 2019-12-27 16:49:01
* description : 
*
============================================*/


#pragma once
#include "elsa.h"
#include "./basicrenderelement.h"
#include "learnopengl.h"

class REBloom_Base : public Renderer::Element
{
public:
    REBloom_Base(const std::string& name="REBloom_Base_unnamed") : Renderer::Element(name) {}

protected: 
    virtual void _PrepareMesh() override;
};


class RECopyTexture : public REBloom_Base
{
public:
    RECopyTexture(const std::string& name="RE_CopyTexture_unnamed") : REBloom_Base(name) {}
    static std::string GetTypeName() { return "RE_CopyTexture"; }
    static std::shared_ptr<RECopyTexture> Create(const std::string& name) { return std::make_shared<RECopyTexture>(name); }
    std::shared_ptr<Renderer::Element> Set(const std::shared_ptr<Texture>& tex) { m_texSource = tex; _Prepare(); return shared_from_this(); }

protected:
    void _PrepareMaterial() override;
    void _PrepareShader() override;

private:
    std::shared_ptr<Texture> m_texSource = nullptr;
};

class REBase : public REBloom_Base
{
public:
    REBase(const std::string& name="RE_Base_unnamed") : REBloom_Base(name) { _Prepare(); }
    static std::string GetTypeName() { return "RE_Base"; }
    static std::shared_ptr<REBase> Create(const std::string& name) { return std::make_shared<REBase>(name); }

protected:
    void _PrepareMaterial() override;
    void _PrepareShader()   override;
private:
    int m_shaderID = 0;
};

class REBright : public REBloom_Base
{
public:
    REBright(const std::string& name="RE_Bright_unnamed") : REBloom_Base(name) { _Prepare(); }
    static std::string GetTypeName() { return "RE_Bright"; }
    static std::shared_ptr<REBright> Create(const std::string& name) { return std::make_shared<REBright>(name); }
protected:
    void _PrepareMaterial() override;
    void _PrepareShader()   override;
};

class REBlur : public REBloom_Base
{
public:
    REBlur(const std::string& name="RE_Bright_unnamed") : REBloom_Base(name) {}
    static std::string GetTypeName() { return "RE_Blur"; }
protected:
    void _PrepareShader()   override;
};

class REBlurH : public REBlur
{
public:
    REBlurH(const std::string& name="RE_BlurH_unnamed") : REBlur(name) { _Prepare(); }
    static std::string GetTypeName() { return "RE_BlurH"; }
    static std::shared_ptr<REBlurH> Create(const std::string& name) { return std::make_shared<REBlurH>(name); }
protected:
    void _PrepareMaterial() override;
};

class REBlurV : public REBlur
{
public:
    REBlurV(const std::string& name="RE_BlurV_unnamed") : REBlur(name) { _Prepare(); }
    static std::string GetTypeName() { return "RE_BlurV"; }
    static std::shared_ptr<REBlurV> Create(const std::string& name) { return std::make_shared<REBlurV>(name); }

protected:
    void _PrepareMaterial() override;
};

class REBloom : public REBloom_Base
{
public:
    REBloom(const std::string& name="REBloom_unnamed") : REBloom_Base(name) {}
    static std::string GetTypeName() { return "RE_Bloom"; }
    static std::shared_ptr<REBloom> Create(const std::string& name) { return std::make_shared<REBloom>(name); }
    std::shared_ptr<Renderer::Element> Set(const std::shared_ptr<Texture>& base, const std::shared_ptr<Texture>& blur);

protected:
    void _PrepareMaterial() override;
    void _PrepareShader()   override;

private:
    std::shared_ptr<Texture> m_base = nullptr;
    std::shared_ptr<Texture> m_blur = nullptr;
};

class Bloom : public LearnOpenGL
{
public:
    Bloom();
    ~Bloom() = default;

    virtual void OnUpdate() override;
    virtual void OnEvent(Event& e) override;
    virtual void OnImgui() override;

private:
    void _Imgui_Viewport();
    void _Imgui_PostProcess();
    void _Imgui_Blur();
    void _Imgui_Bloom();
    void _Imgui_Containers();
    void _Imgui_GroundPlane();


protected:
    void _UpdateUniform();

private:
    void _RenderToTexture_BaseBright();
    void _RenderToTexture_Blur();
    void _RenderToTexture_Bloom();

    void _RenderToScreen(const std::shared_ptr<Viewport>& vp, const std::shared_ptr<Renderer::Element>& ele);


private:
    bool _OnWindowResizeEvent(WindowResizeEvent& e);

private:
    bool m_splitViewport = true;
    std::shared_ptr<Viewport> m_vpBase      = Viewport::Create("vp_Base"     )->SetRange(0.0, 0.5, 0.5, 0.5);
    std::shared_ptr<Viewport> m_vpBright    = Viewport::Create("vp_Bright"   )->SetRange(0.5, 0.5, 0.5, 0.5);
    std::shared_ptr<Viewport> m_vpBlur      = Viewport::Create("vp_Blur"     )->SetRange(0.0, 0.0, 0.5, 0.5);
    std::shared_ptr<Viewport> m_vpBloom     = Viewport::Create("vp_Bloom"    )->SetRange(0.5, 0.0, 0.5, 0.5);
    std::shared_ptr<Viewport> m_vpOffscreen = Viewport::Create("vp_Offscreen")->SetRange(0.0, 0.0, 0.5, 0.5);
    std::shared_ptr<Viewport> m_vpCurrent   = m_vpBase;

    const int m_width  = 1920;
    const int m_height = 1080;
    std::shared_ptr<RenderBuffer> m_rbDepthStencil  = RenderBuffer::Create(m_width, m_height, RenderBuffer::Format::DEPTH24_STENCIL8, 1, "rb_DepthStencil");
    std::shared_ptr<Texture> m_texBase     = Renderer::Resources::Create<Texture2D>("t2d_Base"    )->Set(m_width, m_height, Texture::Format::RGB16F, 1);
    std::shared_ptr<Texture> m_texBright   = Renderer::Resources::Create<Texture2D>("t2d_Bright"  )->Set(m_width, m_height, Texture::Format::RGB16F, 1);
    std::shared_ptr<Texture> m_texBlurPing = Renderer::Resources::Create<Texture2D>("t2d_BlurPing")->Set(m_width, m_height, Texture::Format::RGB16F, 1);
    std::shared_ptr<Texture> m_texBlurPong = Renderer::Resources::Create<Texture2D>("t2d_BlurPong")->Set(m_width, m_height, Texture::Format::RGB16F, 1);
    std::shared_ptr<Texture> m_texBloom    = m_texBlurPing;

    std::shared_ptr<FrameBuffer> m_fbBaseBright = FrameBuffer::Create(m_width, m_height)->AddColorBuffer("cb_Base"    , m_texBase    )->AddColorBuffer("cb_Bright", m_texBright)->AddRenderBuffer("rb_DepthStencil", m_rbDepthStencil);
    std::shared_ptr<FrameBuffer> m_fbBlurPing   = FrameBuffer::Create(m_width, m_height)->AddColorBuffer("cb_BlurPing", m_texBlurPing);
    std::shared_ptr<FrameBuffer> m_fbBlurPong   = FrameBuffer::Create(m_width, m_height)->AddColorBuffer("cb_BlurPong", m_texBlurPong);
    std::shared_ptr<FrameBuffer> m_fbBloom      = FrameBuffer::Create(m_width, m_height)->AddColorBuffer("cb_Bloom"   , m_texBloom   );

    int m_blurIteration = 2;
    std::shared_ptr<Renderer::Element> m_eleBase        = Renderer::Resources::Create<REBase       >("ele_Base"       );
    std::shared_ptr<Renderer::Element> m_eleBright      = Renderer::Resources::Create<REBright     >("ele_Bright"     );
    std::shared_ptr<Renderer::Element> m_eleBlurH       = Renderer::Resources::Create<REBlurH      >("ele_BlurH"      );
    std::shared_ptr<Renderer::Element> m_eleBlurV       = Renderer::Resources::Create<REBlurV      >("ele_BlurV"      );
    std::shared_ptr<Renderer::Element> m_eleBloom       = Renderer::Resources::Create<REBloom      >("ele_Bloom"      )->Set(m_texBase, m_texBlurPong);
    std::shared_ptr<Renderer::Element> m_eleSkybox      = Renderer::Resources::Create<RESkybox     >("ele_Skybox"     );
    std::shared_ptr<Renderer::Element> m_eleContainers  = Renderer::Resources::Create<REContainers >("ele_Containers" )->Set(2000, 50, 20);
    std::shared_ptr<Renderer::Element> m_eleGroundPlane = Renderer::Resources::Create<REGroundPlane>("ele_GroundPlane");
    std::shared_ptr<Renderer::Element> m_eleCopyTexture = Renderer::Resources::Create<RECopyTexture>("ele_CopyTexture")->Set(m_texBright);
    std::shared_ptr<Renderer::Element> m_eleCurrent     = m_eleBase;
};

