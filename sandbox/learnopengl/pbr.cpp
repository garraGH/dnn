/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : sandbox/learnopengl/pbr.cpp
* author      : Garra
* time        : 2019-12-22 11:58:31
* description : 
*
============================================*/


#include "stb_image.h"
#include "pbr.h"
#include "glm/gtc/type_ptr.hpp"
#include "osdialog.h"

PBR::PBR()
{
    PROFILE_FUNCTION
    _IrradianceConvolution();
    _PrefilterConvolution();
    _BRDF();
}

PBR::~PBR()
{

}

void PBR::OnUpdate()
{
    PROFILE_FUNCTION
    _UpdateUniform();
    _Render();
}

void PBR::_Render()
{
    Renderer::BeginScene(m_viewportMain);
    Renderer::Submit(m_eleCubebox);
    Renderer::Submit(m_eleQuad);
    Renderer::Submit(m_eleCubeboxcross);
    Renderer::Submit(m_eleSpheres);
    Renderer::Submit(m_eleSkybox);
    Renderer::EndScene();
}

void PBR::OnEvent(Event& e)
{
    m_viewportMain->OnEvent(e);
}

void PBR::OnImgui()
{
    ImGui::Begin("PBR");

    if(m_eleSkybox->OnImgui())
    {
        _IrradianceConvolution();
        _PrefilterConvolution();
    }

    m_eleCubeboxcross->OnImgui();
    m_eleSpheres->OnImgui();
    m_eleCubebox->OnImgui();

    m_viewportMain->OnImGuiRender();
    ImGui::End();
}




void PBR::_UpdateUniform()
{
    using MU = Material::Uniform;
    const std::shared_ptr<Camera>& cam = m_viewportMain->GetCamera();
    Renderer::Resources::Get<UniformBuffer>("ub_Transform")->Upload("WS2CS", glm::value_ptr(cam->World2Clip()));
    glm::mat4 ws2vs = cam->World2View();
    ws2vs = glm::mat4(glm::mat3(ws2vs));
    glm::mat4 vs2cs = cam->View2Clip();
    Renderer::Resources::Get<MU>("mu_WS2VS")->UpdateData(&ws2vs);
    Renderer::Resources::Get<MU>("mu_VS2CS")->UpdateData(&vs2cs);
    Renderer::Resources::Get<MU>("mu_CameraPosition")->UpdateData(glm::value_ptr(cam->GetPosition()));
}

glm::mat4 PBR::_CaptureView(int i)
{
    switch(i)
    {
        case 0:  return glm::lookAt(glm::vec3(0), glm::vec3(+1, 0, 0), glm::vec3(0, -1, 0));
        case 1:  return glm::lookAt(glm::vec3(0), glm::vec3(-1, 0, 0), glm::vec3(0, -1, 0));
        case 2:  return glm::lookAt(glm::vec3(0), glm::vec3(0, +1, 0), glm::vec3(0, 0, +1));
        case 3:  return glm::lookAt(glm::vec3(0), glm::vec3(0, -1, 0), glm::vec3(0, 0, -1));
        case 4:  return glm::lookAt(glm::vec3(0), glm::vec3(0, 0, +1), glm::vec3(0, -1, 0));
        default: return glm::lookAt(glm::vec3(0), glm::vec3(0, 0, -1), glm::vec3(0, -1, 0));
    }
}

void PBR::_IrradianceConvolution()
{
    PROFILE_FUNCTION
    static bool bFirst = true;
    if(bFirst)
    {
        const unsigned int w = 32;
        const unsigned int h = 32;
        Renderer::Resources::Create<Viewport>("vp_IrradianceConvolution")->SetRange(0, 0, w, h);

        std::shared_ptr<Texture> tcmIrradiance = Renderer::Resources::Get<TextureCubemap>("tcm_Irradiance");
        tcmIrradiance->Set(w, h, Texture::Format::RGB16F);
        std::shared_ptr<FrameBuffer> fbCapture = Renderer::Resources::Create<FrameBuffer>("fb_IrradianceConvolution")->Set(w, h);
        fbCapture->AddRenderBuffer("rb_DepthStencil", RenderBuffer::Format::DEPTH_COMPONENT24);
        fbCapture->AddCubemapBuffer("cmb_Irradiance", tcmIrradiance);

        using MU = Material::Uniform;
        glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
        std::shared_ptr<MU> muProjection = Renderer::Resources::Exist<MU>("mu_Projection")? Renderer::Resources::Get<MU>("mu_Projection") : Renderer::Resources::Create<MU>("mu_Projection")->Set(MU::Type::Mat4x4, 1, glm::value_ptr(captureProjection));
        std::shared_ptr<MU> muView = Renderer::Resources::Exist<MU>("mu_View")? Renderer::Resources::Get<MU>("mu_View") : Renderer::Resources::Create<MU>("mu_View")->SetType(MU::Type::Mat4x4);
        std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("mtr_IrradianceConvolution");
        mtr->SetUniform("u_VS2CS", muProjection);
        mtr->SetUniform("u_WS2VS", muView);
        mtr->SetTexture("u_EnvironmentMap", Renderer::Resources::Get<TextureCubemap>("tcm_Skybox"));
        std::shared_ptr<Shader> shader = Renderer::Resources::Create<Shader>("shader_IrradianceConvolution")->LoadFromFile("/home/garra/study/dnn/assets/shader/IrradianceConvolution.glsl");
        Renderer::Resources::Create<Renderer::Element>("ele_IrradianceConvolution")->Set(Renderer::Resources::Get<Elsa::Mesh>("mesh_Skybox"), mtr, shader);
        bFirst = false;
    }

    std::shared_ptr<Viewport> vpCapture = Renderer::Resources::Get<Viewport>("vp_IrradianceConvolution");
    std::shared_ptr<FrameBuffer> fbCapture = Renderer::Resources::Get<FrameBuffer>("fb_IrradianceConvolution");
    std::shared_ptr<Material::Uniform> muView = Renderer::Resources::Get<Material::Uniform>("mu_View");
    for(int i=0; i<6; i++)
    {
        fbCapture->UseCubemapFace("cmb_Irradiance", TextureCubemap::Face(i));
        muView->UpdateData(glm::value_ptr(_CaptureView(i)));
        Renderer::BeginScene(vpCapture, fbCapture);
        Renderer::Submit("ele_IrradianceConvolution");
        Renderer::EndScene();
    }
}  

void PBR::_PrefilterConvolution()
{
    PROFILE_FUNCTION
    const unsigned int w = 128;
    const unsigned int h = 128;
    const unsigned int maxMipLevels = 5;
    static bool bFirst = true;
    
    using MU = Material::Uniform;
    if(bFirst)
    {
        Renderer::Resources::Create<Viewport>("vp_Prefilter");

        std::shared_ptr<Texture> tcmPrefilter = Renderer::Resources::Get<TextureCubemap>("tcm_Prefilter");
        tcmPrefilter->Set(w, h, Texture::Format::RGB16F, 1, maxMipLevels);

        std::shared_ptr<FrameBuffer> fbCapture = Renderer::Resources::Create<FrameBuffer>("fb_Prefilter")->Set(w, h);
        fbCapture->AddRenderBuffer("rb_DepthStencil", RenderBuffer::Format::DEPTH_COMPONENT24);
        fbCapture->AddCubemapBuffer("cmb_Prefilter", tcmPrefilter);

        std::shared_ptr<Texture> tcmSkybox = Renderer::Resources::Get<TextureCubemap>("tcm_Skybox");
        float resolution = (float)tcmSkybox->GetWidth();
        glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
        std::shared_ptr<MU> muProjection = Renderer::Resources::Exist<MU>("mu_Projection")? Renderer::Resources::Get<MU>("mu_Projection") : Renderer::Resources::Create<MU>("mu_Projection")->Set(MU::Type::Mat4x4, 1, glm::value_ptr(captureProjection));
        std::shared_ptr<MU> muView = Renderer::Resources::Exist<MU>("mu_View")? Renderer::Resources::Get<MU>("mu_View") : Renderer::Resources::Create<MU>("mu_View")->SetType(MU::Type::Mat4x4);
        std::shared_ptr<MU> muRoughness = Renderer::Resources::Create<MU>("mu_Roughness")->SetType(MU::Type::Float1);
        std::shared_ptr<MU> muResolution = Renderer::Resources::Create<MU>("mu_Resolution")->Set(MU::Type::Float1, 1, &resolution);
        std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("mtr_Prefilter");
        mtr->SetUniform("u_VS2CS", muProjection);
        mtr->SetUniform("u_WS2VS", muView);
        mtr->SetTexture("u_EnvironmentMap", tcmSkybox);
        mtr->SetUniform("u_Roughness", muRoughness);
        mtr->SetUniform("u_Resolution", muResolution);

        std::shared_ptr<Elsa::Mesh> mesh = Renderer::Resources::Get<Elsa::Mesh>("mesh_Skybox");
        std::shared_ptr<Shader> shader = Renderer::Resources::Create<Shader>("shader_Prefilter")->LoadFromFile("/home/garra/study/dnn/assets/shader/PrefilterConvolution.glsl");
        Renderer::Resources::Create<Renderer::Element>("ele_Prefilter")->Set(mesh, mtr, shader);
        bFirst = false;
    }

    std::shared_ptr<Viewport> vpCapture = Renderer::Resources::Get<Viewport>("vp_Prefilter");
    std::shared_ptr<FrameBuffer> fbCapture = Renderer::Resources::Get<FrameBuffer>("fb_Prefilter");
    std::shared_ptr<MU> muView = Renderer::Resources::Get<MU>("mu_View");
    std::shared_ptr<MU> muRoughness = Renderer::Resources::Get<MU>("mu_Roughness");
    float roughness = 0;
    for(unsigned int mip=0; mip<maxMipLevels; mip++)
    {
        unsigned int mipWidth  = w*std::pow(0.5, mip);
        unsigned int mipHeight = h*std::pow(0.5, mip);
        vpCapture->SetRange(0, 0, mipWidth, mipHeight);
        roughness = (float)mip/(float)(maxMipLevels-1);
        muRoughness->UpdateData(&roughness);
        for(int i=0; i<6; i++)
        {
            fbCapture->UseCubemapFace("cmb_Prefilter", TextureCubemap::Face(i), mip);
            muView->UpdateData(glm::value_ptr(_CaptureView(i)));
            Renderer::BeginScene(vpCapture, fbCapture);
            Renderer::Submit("ele_Prefilter");
            Renderer::EndScene();
        }
    }
}


void PBR::_BRDF()
{
    PROFILE_FUNCTION
    const unsigned int w = 512;
    const unsigned int h = 512;
    static bool bFirst = true;
    if(bFirst)
    {
        std::shared_ptr<Texture> t2dBRDF = Renderer::Resources::Get<Texture2D>("t2d_BRDF");
        t2dBRDF->Set(w, h, Texture::Format::RG16F);
        Renderer::Resources::Create<Viewport>("vp_BRDF")->SetRange(0, 0, w, h);
        std::shared_ptr<FrameBuffer> fbBRDF = Renderer::Resources::Create<FrameBuffer>("fb_BRDF");
        fbBRDF->AddRenderBuffer("rb_DepthStencil", RenderBuffer::Format::DEPTH_COMPONENT24);
        fbBRDF->AddColorBuffer("cb_BRDF", t2dBRDF);
        Renderer::Resources::Create<Shader>("shader_BRDF")->LoadFromFile("/home/garra/study/dnn/assets/shader/BRDF.glsl");
        Renderer::Resources::Create<Renderer::Element>("ele_BRDF")->SetMesh(Renderer::Resources::Get<Elsa::Mesh>("mesh_Quad"));
        bFirst = false;
    }

    std::shared_ptr<Viewport> vpBRDF = Renderer::Resources::Get<Viewport>("vp_BRDF");
    std::shared_ptr<FrameBuffer> fbBRDF = Renderer::Resources::Get<FrameBuffer>("fb_BRDF");
    Renderer::BeginScene(vpBRDF, fbBRDF);
    Renderer::Submit("ele_BRDF", "shader_BRDF");
    Renderer::EndScene();
}

