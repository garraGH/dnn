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
    enum class MaterialProperty : int
    {
        HasDiffuseReflectance   = 0x01<<0, 
        HasSpecularReflectance  = 0x01<<1, 
        HasEmissiveColor        = 0x01<<2, 
        HasDiffuseMap           = 0x01<<3, 
        HasSpecularMap          = 0x01<<4, 
        HasEmissiveMap          = 0x01<<5, 
        HasNormalMap            = 0x01<<6, 
        HasDepthMap             = 0x01<<7, 
    };

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
    void _UpdateMaterialUniforms();

    void _PrepareUniformBuffers();
    void _UpdateTexture(std::shared_ptr<Texture>& tex);

private:
    std::shared_ptr<Viewport> m_viewport = Viewport::Create("LearnOpenGL_Viewport_Main");
    std::shared_ptr<Model> m_crysisNanoSuit = nullptr;
    std::shared_ptr<Model> m_silkingMachine = nullptr;
    std::shared_ptr<Model> m_horse = nullptr;
    std::shared_ptr<Model> m_trailer = nullptr;
    std::shared_ptr<Model> m_bulb = nullptr;
    std::shared_ptr<Model> m_handLight = nullptr;
    std::shared_ptr<Shader> m_shaderPos = nullptr;
    std::shared_ptr<Shader> m_shaderColor = nullptr;
    std::shared_ptr<Shader> m_shaderBlinnPhong = nullptr;
    std::shared_ptr<Shader> m_shaderOfMaterial = nullptr;
    std::shared_ptr<Renderer::Element> m_unitCubic = nullptr;

    struct 
    {
        glm::vec3* diffuseReflectance = nullptr;
        glm::vec3* specularReflectance = nullptr;
        glm::vec3* emissiveColor = nullptr;
        float* shininess = nullptr;
        float* depthScale = nullptr;
        std::shared_ptr<Texture> diffuseMap = nullptr;
        std::shared_ptr<Texture> specularMap = nullptr;
        std::shared_ptr<Texture> emissiveMap = nullptr;
        std::shared_ptr<Texture> normalMap = nullptr;
        std::shared_ptr<Texture> depthMap = nullptr;
        bool hasDiffuseReflectance = true;
        bool hasSpecularReflectance = true;
        bool hasEmissiveColor = true;
        bool hasDiffuseMap = true;
        bool hasSpecularMap = true;
        bool hasEmissiveMap = true;
        bool hasNormalMap = true;
        bool hasDepthMap = true;
    }
    m_material; 
    int m_shaderID = 0b11111111;
    void _UpdateShaderID();
    void _AddMaterialProperty(MaterialProperty mp);
    void _RemoveMaterialProperty(MaterialProperty mp);
    std::string _StringOfShaderID() const;
    std::string _MacrosOfShaderID() const;

    struct DirectionalLight
    {
        glm::vec4 clr;
        glm::vec4 dir;
    };

    struct PointLight
    {
        glm::vec4 clr;
        glm::vec4 pos;
        glm::vec4 coe;
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
        float degOuterCone;
    };

    DirectionalLight m_dLight = { glm::vec4(1.0f), glm::vec4(0, 0, -1, 0)};
    PointLight m_pLight = { glm::vec4(1, 0, 0, 1), glm::vec4(0, 5, 0, 1), glm::vec4(1.0, 0.09, 0.032, 0.0) };
    SpotLight m_sLight = { glm::vec3(0, 1, 0), std::cos(glm::radians(15.0f)), glm::vec3(2, 0, 0), std::cos(glm::radians(20.0f)), glm::vec3(-1, 0, 0), 15, glm::vec3(1.0, 0.22, 0.20), 20 };
    SpotLight m_fLight = { glm::vec3(0, 1, 0), std::cos(glm::radians(15.0f)), glm::vec3(2, 0, 0), std::cos(glm::radians(20.0f)), glm::vec3(-1, 0, 0), 15, glm::vec3(1.0, 0.22, 0.20), 20 };

    glm::vec3* m_ambientColor = nullptr;
    glm::vec2* m_rightTopTexCoord = nullptr;

    bool m_showSky = true;
    bool m_showGround= false;
    
    PostProcess m_pp = PostProcess::None;

    const unsigned int m_numOfInstance = 2000;

    unsigned int m_samples = 4;
    std::shared_ptr<FrameBuffer> m_fbSS = FrameBuffer::Create(1920, 1080, 1);  // framebufferSingleSample
    std::shared_ptr<FrameBuffer> m_fbMS = FrameBuffer::Create(1920, 1080, m_samples);  // framebufferMultiSample
};

