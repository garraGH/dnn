/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : sandbox/learnopengl/bloom.cpp
* author      : Garra
* time        : 2019-12-27 16:48:59
* description : 
*
============================================*/


#include "bloom.h"
#include "glm/gtc/type_ptr.hpp"

///////////////////////////////////////////////
void REBloom_Base::_PrepareMesh()
{
    if(Renderer::Resources::Exist<Elsa::Mesh>("mesh_Screen"))
    {
        m_mesh = Renderer::Resources::Get<Elsa::Mesh>("mesh_Screen");
        return;
    }

    float vertices[] = 
    {
        -1, -1, 
        +1, -1, 
        +1, +1,
        -1, +1, 
    };
    unsigned char indices[] = 
    { 
        0, 1, 2, 
        0, 2, 3,
    };

    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout( {{Buffer::Element::DataType::Float2, "a_Position", false}} );
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout( {{Buffer::Element::DataType::UChar}} );
    m_mesh= Renderer::Resources::Create<Elsa::Mesh>("mesh_Screen")->Set(ib, {vb});
}

///////////////////////////////////////////////
void RECopyTexture::_PrepareMaterial()
{
    using MU = Material::Uniform;
    std::shared_ptr<MU> muRightTopTexCoord   = Renderer::Resources::Create<MU>("mu_RightTopTexCoord"  )->SetType(MU::Type::Float2);
    std::shared_ptr<MU> muLeftBottomTexCoord = Renderer::Resources::Create<MU>("mu_LeftBottomTexCoord")->SetType(MU::Type::Float2);
    m_material = Renderer::Resources::Create<Material>("mtr_CopyTexture");
    m_material->SetTexture("u_Texture", m_texSource);
    m_material->SetUniform("u_RightTopTexCoord"  , muRightTopTexCoord  );
    m_material->SetUniform("u_LeftBottomTexCoord", muLeftBottomTexCoord);
}

void RECopyTexture::_PrepareShader()
{
    m_shader = Renderer::Resources::Create<Shader>("shader_CopyTexture")->LoadFromFile("/home/garra/study/dnn/assets/shader/CopyTexture.glsl");
}

///////////////////////////////////////////////
void REBase::_PrepareMaterial()
{
    int postProcess = 0;
    using MU = Material::Uniform;
    std::shared_ptr<MU> muGamma              = Renderer::Resources::Create<MU>("mu_Gamma"             )->SetType(MU::Type::Float1);
    std::shared_ptr<MU> muExposure           = Renderer::Resources::Create<MU>("mu_Exposure"          )->SetType(MU::Type::Float1);
    std::shared_ptr<MU> muPostProcess        = Renderer::Resources::Create<MU>("mu_PostProcess"       )->Set(MU::Type::Int1, 1, &postProcess);
    std::shared_ptr<MU> muRightTopTexCoord   = Renderer::Resources::Create<MU>("mu_RightTopTexCoord"  )->SetType(MU::Type::Float2);
    std::shared_ptr<MU> muLeftBottomTexCoord = Renderer::Resources::Create<MU>("mu_LeftBottomTexCoord")->SetType(MU::Type::Float2);

    m_material = Renderer::Resources::Create<Material>("mtr_Base");
    m_material->SetUniform("u_Gamma"             , muGamma             );
    m_material->SetUniform("u_Exposure"          , muExposure          );
    m_material->SetUniform("u_PostProcess"       , muPostProcess       );
    m_material->SetUniform("u_RightTopTexCoord"  , muRightTopTexCoord  );
    m_material->SetUniform("u_LeftBottomTexCoord", muLeftBottomTexCoord);

    m_material->SetTexture("u_Offscreen", Renderer::Resources::Get<Texture2D>("t2d_Base"));
}

void REBase::_PrepareShader()
{
    m_shader = Renderer::Resources::Create<Shader>("shader_Base")->LoadFromFile("/home/garra/study/dnn/assets/shader/OffscreenTexture.glsl");
}

//////////////////////////////////////////////////
void REBright::_PrepareMaterial()
{
    m_material = Renderer::Resources::Create<Material>("mtr_Bright");
    m_material->SetTexture("u_Offscreen", Renderer::Resources::Get<Texture2D>("t2d_Bright"));
    m_material->SetUniform("u_RightTopTexCoord"  , Renderer::Resources::Get<Material::Uniform>("mu_RightTopTexCoord"));
    m_material->SetUniform("u_LeftBottomTexCoord", Renderer::Resources::Get<Material::Uniform>("mu_LeftBottomTexCoord"));
}

void REBright::_PrepareShader()
{
    m_shader = Renderer::Resources::Create<Shader>("shader_Bright")->LoadFromFile("/home/garra/study/dnn/assets/shader/OffscreenTexture.glsl");
}

//////////////////////////////////////////////////
void REBlur::_PrepareShader()
{
    m_shader = Renderer::Resources::Create<Shader>("Blur")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blur.glsl");
}

void REBlurH::_PrepareMaterial()
{
    m_material = Renderer::Resources::Create<Material>("mtr_BlurH");
    using MU = Material::Uniform;
    int direction = 1;
    std::shared_ptr<MU> muHorizontalBlur = Renderer::Resources::Create<MU>("mu_HorizontalBlur")->Set(MU::Type::Int1, 1, &direction);

    m_material->SetUniform("u_LeftBottomTexCoord", Renderer::Resources::Get<MU>("mu_LeftBottomTexCoord"));
    m_material->SetUniform("u_RightTopTexCoord", Renderer::Resources::Get<MU>("mu_RightTopTexCoord"));
    m_material->SetTexture("u_Offscreen", Renderer::Resources::Get<Texture2D>("t2d_BlurPong"));
    m_material->SetUniform("u_Direction", muHorizontalBlur);
}

void REBlurV::_PrepareMaterial()
{
    m_material = Renderer::Resources::Create<Material>("mtr_BlurV");
    using MU = Material::Uniform;
    int direction = 0;
    std::shared_ptr<MU> muVerticalBlur = Renderer::Resources::Create<MU>("mu_VerticalBlur")->Set(MU::Type::Int1, 1, &direction);
    m_material->SetUniform("u_LeftBottomTexCoord", Renderer::Resources::Get<MU>("mu_LeftBottomTexCoord")); 
    m_material->SetUniform("u_RightTopTexCoord", Renderer::Resources::Get<MU>("mu_RightTopTexCoord"));
    m_material->SetTexture("u_Offscreen", Renderer::Resources::Get<Texture2D>("t2d_BlurPing"));
    m_material->SetUniform("u_Direction", muVerticalBlur);
}


//////////////////////////////////////////////////
std::shared_ptr<Renderer::Element> REBloom::Set(const std::shared_ptr<Texture>& base, const std::shared_ptr<Texture>& blur)
{
    m_base = base;
    m_blur = blur;
    _Prepare();
    return shared_from_this();
}

void REBloom::_PrepareMaterial()
{
    m_material = Renderer::Resources::Create<Material>("mtr_Bloom");
    using MU = Material::Uniform;
    m_material->SetTexture("u_Base", m_base);
    m_material->SetTexture("u_Blur", m_blur);
    m_material->SetUniform("u_RightTopTexCoord"  , Renderer::Resources::Get<MU>("mu_RightTopTexCoord"  ));
    m_material->SetUniform("u_LeftBottomTexCoord", Renderer::Resources::Get<MU>("mu_LeftBottomTexCoord")); 
}

void REBloom::_PrepareShader()
{
    m_shader = Renderer::Resources::Create<Shader>("Bloom")->LoadFromFile("/home/garra/study/dnn/assets/shader/Bloom.glsl");
}
///////////////////////////////////////////////////

Bloom::Bloom()
{
    PROFILE_FUNCTION
    std::shared_ptr<Camera> cam = m_vpOffscreen->GetCamera();
    cam->SetFrameBuffer(m_fbBaseBright);
    cam->SetPosition(glm::vec3(0, 2, 10));
    cam->SetTarget(glm::vec3(0));
    m_vpBase->AttachCamera(cam);
    m_vpBright->AttachCamera(cam);
    m_vpBlur->AttachCamera(cam);
    m_vpBloom->AttachCamera(cam);
}

void Bloom::OnUpdate()
{
    PROFILE_FUNCTION
    _UpdateUniform();

    _RenderToTexture_BaseBright();
    _RenderToTexture_Blur();
    _RenderToTexture_Bloom();

    if(m_splitViewport)
    {
        _RenderToScreen(m_vpBase, m_eleBase);
        _RenderToScreen(m_vpBright, m_eleBright);
        _RenderToScreen(m_vpBlur, m_eleBlurH);
        _RenderToScreen(m_vpBloom, m_eleBloom);
    }
    else
    {
        _RenderToScreen(m_vpCurrent, m_eleCurrent);
    }
}

void Bloom::_UpdateUniform()
{
    using MU = Material::Uniform;
    const std::shared_ptr<Camera>& cam = m_vpOffscreen->GetCamera();
    glm::mat4 ws2vs = cam->World2View();
    ws2vs = glm::mat4(glm::mat3(ws2vs));

    glm::mat4 vs2cs = cam->View2Clip();
    Renderer::Resources::Get<UniformBuffer>("ub_Transform")->Upload("WS2CS", glm::value_ptr(cam->World2Clip()));
    Renderer::Resources::Get<MU>("mu_WS2VS")->UpdateData(&ws2vs);
    Renderer::Resources::Get<MU>("mu_VS2CS")->UpdateData(&vs2cs);
    Renderer::Resources::Get<MU>("mu_CameraPosition")->UpdateData(glm::value_ptr(cam->GetPosition()));
    Renderer::Resources::Get<MU>("mu_NearCorners")->UpdateData(&cam->GetNearCornersInWorldSpace()[0]);
    Renderer::Resources::Get<MU>("mu_FarCorners")->UpdateData(&cam->GetFarCornersInWorldSpace()[0]);

    std::array<float, 4> r = m_vpOffscreen->GetRange();
    glm::vec2 leftBottomTexCoord = glm::vec2(r[0]/m_width, r[1]/m_height);
    glm::vec2 rightTopTexCoord = glm::vec2((r[0]+r[2])/m_width, (r[1]+r[3])/m_height);
    Renderer::Resources::Get<MU>("mu_LeftBottomTexCoord")->UpdateData(glm::value_ptr(leftBottomTexCoord));
    Renderer::Resources::Get<MU>("mu_RightTopTexCoord")->UpdateData(glm::value_ptr(rightTopTexCoord));
}

void Bloom::OnEvent(Event& e)
{
    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(BIND_EVENT_CALLBACK(Bloom, _On##event))
    DISPATCH(WindowResizeEvent);
#undef DISPATCH
    if(m_splitViewport)
    {
        m_vpBase->OnEvent(e);
        m_vpBlur->OnEvent(e);
        m_vpBright->OnEvent(e);
        m_vpBloom->OnEvent(e);
    }
    else
    {
        m_vpCurrent->OnEvent(e);
    }
}

void Bloom::OnImgui()
{
    ImGui::Begin("Bloom");
    _Imgui_Viewport();
    _Imgui_PostProcess();
    _Imgui_Blur();
    _Imgui_Bloom();
    _Imgui_Containers();
    _Imgui_GroundPlane();
    ImGui::End();
}

void Bloom::_Imgui_Viewport()
{
    if(ImGui::CollapsingHeader("VIEWPORT"))
    {
        ImGui::Indent();
        ImGui::Checkbox("SplitViewport", &m_splitViewport);
        if(m_splitViewport)
        {
            m_vpBase     ->SetRange(0.0, 0.5, 0.5, 0.5);
            m_vpBright   ->SetRange(0.5, 0.5, 0.5, 0.5);
            m_vpBlur     ->SetRange(0.0, 0.0, 0.5, 0.5);
            m_vpBloom    ->SetRange(0.5, 0.0, 0.5, 0.5);
            m_vpOffscreen->SetRange(0.0, 0.0, 0.5, 0.5);
        }
        else
        {
            static int e = 0;
            ImGui::SameLine();
            ImGui::RadioButton("Base"  , &e, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Bright", &e, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Blur"  , &e, 2);
            ImGui::SameLine();
            ImGui::RadioButton("Bloom" , &e, 3);
            switch(e)
            {
                case 0:  m_vpCurrent = m_vpBase;    m_eleCurrent = m_eleBase;   break;
                case 1:  m_vpCurrent = m_vpBright;  m_eleCurrent = m_eleBright; break;
                case 2:  m_vpCurrent = m_vpBlur;    m_eleCurrent = m_eleBlurH;  break;
                default: m_vpCurrent = m_vpBloom;   m_eleCurrent = m_eleBloom;  break;
            }
            m_vpCurrent  ->SetRange(0, 0, 1, 1);
            m_vpOffscreen->SetRange(0, 0, 1, 1);
        }
        ImGui::Unindent();
    }
}

void Bloom::_Imgui_PostProcess()
{
    if(ImGui::CollapsingHeader("POSTPROCESS"))
    {
        static int e = 0;
        ImGui::Indent();
        ImGui::RadioButton("None"  , &e, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Gray"  , &e, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Smooth", &e, 2);
        ImGui::SameLine();
        ImGui::RadioButton("Edge"  , &e, 3);
        Renderer::Resources::Get<Material::Uniform>("mu_PostProcess")->UpdateData(&e);
        ImGui::Unindent();
    }
}

void Bloom::_Imgui_Blur()
{
    if(ImGui::CollapsingHeader("BLUR"))
    {
        ImGui::Indent();
        ImGui::DragInt("NumOfIteration", &m_blurIteration, 1, 0, 100);
        ImGui::Unindent();
    }
}

void Bloom::_Imgui_Bloom()
{
    if(ImGui::CollapsingHeader("BLOOM"))
    {
        static float t = 1;
        ImGui::Indent();
        if(ImGui::DragFloat("Threshold", &t, 0.1,  0.1f,  10.0f))
        {
            Renderer::Resources::Get<Material::Uniform>("mu_BloomThreshold")->UpdateData(&t);
        }
        ImGui::Unindent();
    }
}

void Bloom::_Imgui_Containers()
{
    m_eleContainers->OnImgui();
}

void Bloom::_Imgui_GroundPlane()
{
    m_eleGroundPlane->OnImgui();
}

bool Bloom::_OnWindowResizeEvent(WindowResizeEvent& e)
{
    float w = e.GetWidth();
    float h = e.GetHeight();

    if(m_splitViewport)
    {
        w *= 0.5;
        h *= 0.5;
    }

    glm::vec2 rightTopTexCoord = glm::vec2(w/m_width, h/m_height);
    Renderer::Resources::Get<Material::Uniform>("mu_RightTopTexCoord")->UpdateData(glm::value_ptr(rightTopTexCoord));

    m_vpOffscreen->OnEvent(e);
    return false;
}

void Bloom::_RenderToTexture_BaseBright()
{
/*
                m_width
-------------------------------------------
|                                         |
|                                         |
|           OffscreenTexture              |
|                                         |
|                                         |
|                        rightTop         |
|----------------------------|            |m_height
|                            |            |
|                            |            |
|                            |            |
|           Viewport                      |            |
|                            |            |
|                            |            |
|                            |            |
|leftBottom                  |            |
-------------------------------------------
*/
    Renderer::BeginScene(m_vpOffscreen, m_fbBaseBright);
    Renderer::Submit(m_eleContainers);
    Renderer::Submit(m_eleGroundPlane);
    Renderer::Submit(m_eleSkybox);
    Renderer::EndScene();
}


void Bloom::_RenderToTexture_Blur()
{
    Renderer::BeginScene(m_vpOffscreen, m_fbBlurPong);
    Renderer::Submit(m_eleCopyTexture);
    Renderer::EndScene();

    for(int i=0; i<m_blurIteration; i++)
    {
        Renderer::BeginScene(m_vpOffscreen, m_fbBlurPing);
        Renderer::Submit(m_eleBlurH);
        Renderer::EndScene();
        Renderer::BeginScene(m_vpOffscreen, m_fbBlurPong);
        Renderer::Submit(m_eleBlurV);
        Renderer::EndScene();
    }
}

void Bloom::_RenderToTexture_Bloom()
{
    Renderer::BeginScene(m_vpOffscreen, m_fbBloom);
    Renderer::Submit(m_eleBloom);
    Renderer::EndScene();
}

void Bloom::_RenderToScreen(const std::shared_ptr<Viewport>& vp,  const std::shared_ptr<Renderer::Element>& ele)
{
    Renderer::BeginScene(vp);
    Renderer::Submit(ele);
    Renderer::EndScene();
}

