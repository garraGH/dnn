/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/layer_learnopengl.h
* author      : Garra
* time        : 2019-10-27 11:06:14
* description : 
*
============================================*/


#pragma once
#include "elsa.h"

class LearnOpenGLLayer : public Layer
{
public:

    enum class PostProcess
    {
        None, 
        Gray, 
        Smooth, 
        Edge
    };

public:
    LearnOpenGLLayer();

    void OnEvent(Event& e) override;
    void OnUpdate(float deltaTime) override;
    void OnImGuiRender() override;
 
    static std::shared_ptr<LearnOpenGLLayer> Create();

protected:
#define ON(event) bool _On##event(event& e)
    ON(WindowResizeEvent);
#undef ON

protected:
    void _PrepareSkybox();
    void _PrepareGroundPlane();
    void _PrepareOffscreenPlane();
    void _PrepareModel();
    void _PrepareUnitCubic();
    void _PrepareSphere(float radius, int subdivision);
    void _PrepareSphere(float radius, int stacks, int sectors);
    void _PrepareSpheresPBR(float radius, int stacks, int sectors);
    void _Subdivision(std::vector<glm::vec3>& vertices, std::vector<glm::i16vec3>& triangles);
    void _UpdateMaterialUniforms();
    std::pair<std::vector<glm::vec3>, std::vector<glm::i16vec3>> _GenSphere(float radius, int stacks, int sectors);

    void _PrepareUniformBuffers();
    void _UpdateTexture(std::shared_ptr<Texture>& tex);

private:
    void _RenderToTexture_HDR();
    void _RenderToScreen_HDR();
    void _RenderToTexture_Blur();
    void _RenderToScreen_Blur();
    void _RenderToTexture_Bloom();
    void _RenderToScreen_Bloom();

private:
    std::shared_ptr<Model> m_crysisNanoSuit = nullptr;
    std::shared_ptr<Model> m_silkingMachine = nullptr;
    std::shared_ptr<Model> m_horse = nullptr;
    std::shared_ptr<Model> m_trailer = nullptr;
    std::shared_ptr<Model> m_bulb = nullptr;
    std::shared_ptr<Model> m_handLight = nullptr;
    std::shared_ptr<Model> m_planet = nullptr;
    std::shared_ptr<Model> m_rock = nullptr;
    std::shared_ptr<Shader> m_shaderPos = nullptr;
    std::shared_ptr<Shader> m_shaderColor = nullptr;
    std::shared_ptr<Shader> m_shaderBlinnPhong = nullptr;
    std::shared_ptr<Shader> m_shaderOfMaterial = nullptr;
    std::shared_ptr<Shader> m_shaderHDR = nullptr;
    std::shared_ptr<Shader> m_shaderBlur = nullptr;
    std::shared_ptr<Shader> m_shaderBloom = nullptr;
    std::shared_ptr<Shader> m_shaderSphere = nullptr;
    std::shared_ptr<Shader> m_shaderSpherePBR0 = nullptr;
    std::shared_ptr<Renderer::Element> m_eleCubic = nullptr;
    std::shared_ptr<Renderer::Element> m_eleSphere = nullptr;
    std::shared_ptr<Renderer::Element> m_eleSpherePBR0 = nullptr;
    std::shared_ptr<Renderer::Element> m_eleBase = nullptr;
    std::shared_ptr<Renderer::Element> m_eleBright = nullptr;
    std::shared_ptr<Renderer::Element> m_eleBlurH = nullptr;
    std::shared_ptr<Renderer::Element> m_eleBlurV = nullptr;
    std::shared_ptr<Renderer::Element> m_eleBloom = nullptr;

    int m_row = 11;
    int m_col = 11;

    struct 
    {
        glm::vec3* diffuseReflectance = nullptr;
        glm::vec3* specularReflectance = nullptr;
        glm::vec3* emissiveColor = nullptr;
        float* shininess = nullptr;
        float* displacementScale = nullptr;
        float* emissiveIntensity = nullptr;
        std::shared_ptr<Texture> diffuseMap = nullptr;
        std::shared_ptr<Texture> specularMap = nullptr;
        std::shared_ptr<Texture> emissiveMap = nullptr;
        std::shared_ptr<Texture> normalMap = nullptr;
        std::shared_ptr<Texture> displacementMap = nullptr;
        bool hasDiffuseReflectance = true;
        bool hasSpecularReflectance = true;
        bool hasEmissiveColor = true;
        bool hasDiffuseMap = true;
        bool hasSpecularMap = true;
        bool hasEmissiveMap = true;
        bool hasNormalMap = true;
        bool hasDisplacementMap = true;
    }
    m_material; 

    float* m_bloomThreshold = nullptr;
    int m_shaderID = 0b111111110;
    void _UpdateShaderID();
    std::string _StringOfShaderID() const;

    struct
    {
        bool enableToneMap = true;
        bool enableGammaCorrection = true;
        float* gamma = nullptr;
        float* exposure = nullptr;
    }
    m_material_HDR;
    int m_shaderID_HDR = 0b11000000000000000;
    void _UpdateShaderID_HDR();
    std::string _StringOfShaderID_HDR() const;

    struct DirectionalLight
    {
        glm::vec3 clr;
        float padding0;
        glm::vec3 dir;
        float intensity;
    };

    struct PointLight
    {
        glm::vec3 clr;
        float padding0;
        glm::vec3 pos;
        float padding1;
        glm::vec3 coe;
        float intensity;
    };

    struct SpotLight
    {
        glm::vec3 clr;
        float cosInnerCone;
        glm::vec3 pos;
        float cosOuterCone;
        glm::vec3 dir;
        float degInnerCone;
        glm::vec3 coe;
        float intensity;
        float degOuterCone;
    };

    DirectionalLight m_dLight = { glm::vec3(1.0f), 0, glm::vec3(0, 0, -1), 1.0f};
    PointLight m_pLight = { glm::vec3(1, 0, 0), 0, glm::vec3(0, 5, 0), 0, glm::vec3(1.0, 0.09, 0.032), 1.0f };
    SpotLight m_sLight = { glm::vec3(0, 1, 0), std::cos(glm::radians(15.0f)), glm::vec3(2, 0, 0), std::cos(glm::radians(20.0f)), glm::vec3(-1, 0, 0), 15, glm::vec3(1.0, 0.22, 0.20), 1.0f, 20 };
    SpotLight m_fLight = { glm::vec3(0, 1, 0), std::cos(glm::radians(15.0f)), glm::vec3(2, 0, 0), std::cos(glm::radians(20.0f)), glm::vec3(-1, 0, 0), 15, glm::vec3(1.0, 0.22, 0.20), 1.0f, 20 };


    glm::vec3* m_ambientColor = nullptr;
    glm::vec2* m_leftBottomTexCoord = nullptr;
    glm::vec2* m_rightTopTexCoord = nullptr;

    bool m_showSky = true;
    bool m_showGround= false;
    
    PostProcess m_pp = PostProcess::None;

    const unsigned int m_numOfInstance = 2000;
    const unsigned int m_numOfLights = 2000;

//     unsigned int m_samples = 1;
//     
// #define WIDTH 1000
// #define HEIGHT 1000
//     std::shared_ptr<FrameBuffer> m_fbSS = FrameBuffer::Create(WIDTH, HEIGHT, 1);           // framebufferSingleSample
//     std::shared_ptr<FrameBuffer> m_fbMS = FrameBuffer::Create(WIDTH, HEIGHT, m_samples);   // framebufferMultiSample
//     std::shared_ptr<FrameBuffer> m_fbBlurH = FrameBuffer::Create(WIDTH, HEIGHT);
//     std::shared_ptr<FrameBuffer> m_fbBlurV = FrameBuffer::Create(WIDTH, HEIGHT);
//     std::shared_ptr<FrameBuffer> m_fbBloom = FrameBuffer::Create(WIDTH, HEIGHT);

    std::shared_ptr<Viewport> m_vpBase = Viewport::Create("Base")->SetRange(0, 0.5, 0.5, 0.5);
    std::shared_ptr<Viewport> m_vpBright = Viewport::Create("Bright")->SetRange(0.5, 0.5, 0.5, 0.5);
    std::shared_ptr<Viewport> m_vpBlur = Viewport::Create("Blur")->SetRange(0, 0, 0.5, 0.5);
    std::shared_ptr<Viewport> m_vpBloom = Viewport::Create("Bloom")->SetRange(0.5, 0, 0.5, 0.5);
    std::shared_ptr<Viewport> m_vpOffscreen = Viewport::Create("Offscreen")->SetRange(0, 0, 0.5, 0.5);
//     std::shared_ptr<Viewport> m_vpBloom = Viewport::Create("Bloom")->SetRange(0, 0, 1.0, 1.0);
//     std::shared_ptr<Viewport> m_vpOffscreen = Viewport::Create("Offscreen")->SetRange(0, 0, 1.0, 1.0);

    const glm::vec2 m_offscreenBufferSize = glm::vec2(1920, 1080);
    std::shared_ptr<RenderBuffer> m_rbDepthStencil = RenderBuffer::Create(m_offscreenBufferSize.x, m_offscreenBufferSize.y, 1, RenderBuffer::Format::DEPTH24_STENCIL8, "HDR_DS");
    std::shared_ptr<Texture> m_texOffscreenBasic = Texture2D::Create("Basic")->Set(m_offscreenBufferSize.x, m_offscreenBufferSize.y, 1, Texture::Format::RGB16F);
    std::shared_ptr<Texture> m_texOffscreenBright = Texture2D::Create("Bright")->Set(m_offscreenBufferSize.x, m_offscreenBufferSize.y, 1, Texture::Format::RGB16F);
    std::shared_ptr<Texture> m_texOffscreenBlurPing = Texture2D::Create("Blur")->Set(m_offscreenBufferSize.x, m_offscreenBufferSize.y, 1, Texture::Format::RGB16F);
    std::shared_ptr<Texture> m_texOffscreenBlurPong = m_texOffscreenBright;
    std::shared_ptr<Texture> m_texOffscreenBloom = m_texOffscreenBlurPing;

    std::shared_ptr<FrameBuffer> m_fbOffscreenHDR = FrameBuffer::Create(m_offscreenBufferSize.x, m_offscreenBufferSize.y)->AddColorBuffer("Basic", m_texOffscreenBasic)->AddColorBuffer("Bright", m_texOffscreenBright)->AddRenderBuffer("DS", m_rbDepthStencil);
    std::shared_ptr<FrameBuffer> m_fbOffscreenBlurPing = FrameBuffer::Create(m_offscreenBufferSize.x, m_offscreenBufferSize.y)->AddColorBuffer("BlurPing", m_texOffscreenBlurPing);
    std::shared_ptr<FrameBuffer> m_fbOffscreenBlurPong = FrameBuffer::Create(m_offscreenBufferSize.x, m_offscreenBufferSize.y)->AddColorBuffer("BlurPong", m_texOffscreenBlurPong);
    std::shared_ptr<FrameBuffer> m_fbOffscreenBloom = FrameBuffer::Create(m_offscreenBufferSize.x, m_offscreenBufferSize.y)->AddColorBuffer("Bloom", m_texOffscreenBloom);

    int m_blurIteration = 2;
    
    bool m_splitViewport = true;
};

