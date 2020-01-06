/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/learnopengl/basicrenderelement.h
* author      : Garra
* time        : 2019-12-30 17:13:50
* description : 
*
============================================*/


#pragma once
#include "elsa.h"
 
class RESkybox : public Renderer::Element
{
public:
    RESkybox(const std::string& name="RE_Skybox_Unnamed");
    static std::string GetTypeName() { return "RE_Skybox"; }
    static std::shared_ptr<RESkybox> Create(const std::string& name) { return std::make_shared<RESkybox>(name); }

    bool OnImgui() override;

protected:
    void _PrepareMesh()     override;
    void _PrepareTexture()  override;
    void _PrepareMaterial() override;
    void _PrepareShader()   override;

private:
    bool _Reload(const char* filename);
    void _GenCubemapFromEquirectangle(const char* filename);
    std::shared_ptr<Texture> m_cubemap = nullptr;
};

class REQuad : public Renderer::Element
{
public:
    REQuad(const std::string& name="RE_Quad_Unnamed");
    static std::string GetTypeName() { return "RE_Quad"; }
    static std::shared_ptr<REQuad> Create(const std::string& name) { return std::make_shared<REQuad>(name); }

protected:
    void _PrepareMesh()     override;
    void _PrepareTexture()  override;
    void _PrepareMaterial() override;
    void _PrepareShader()   override;
};

class RECubebox : public Renderer::Element
{
public:
    RECubebox(const std::string& name="RE_Cubebox_Unnamed");
    static std::string GetTypeName() { return "RE_Cubebox"; }
    static std::shared_ptr<RECubebox> Create(const std::string& name) { return std::make_shared<RECubebox>(name); }

    bool OnImgui() override;

protected:
    void _PrepareMesh()     override;
    void _PrepareTexture()  override;
    void _PrepareMaterial() override;
    void _PrepareShader()   override;

private:
    float* m_refractiveIndex = nullptr;
};


class RECubeboxCross : public Renderer::Element
{
public:
    RECubeboxCross(const std::string& name="RE_CubeboxCross_Unnamed");
    static std::string GetTypeName() { return "RE_CubeboxCross"; }
    static std::shared_ptr<RECubeboxCross> Create(const std::string& name) { return std::make_shared<RECubeboxCross>(name); }

    bool OnImgui() override;

protected:
    void _PrepareMesh()     override;
    void _PrepareTexture()  override;
    void _PrepareMaterial() override;
    void _PrepareShader()   override;
};

class RESpheres : public Renderer::Element
{
public:
    RESpheres(const std::string& name="RE_Spheres_Unnamed");
    std::shared_ptr<Renderer::Element> Set(int row, int col, float spacing, float radius, int stacks, int sectors);
    static std::string GetTypeName() { return "RE_Spheres"; }
    static std::shared_ptr<RESpheres> Create(const std::string& name) { return std::make_shared<RESpheres>(name); }
    bool OnImgui() override;

protected:
    void _PrepareMesh()     override;
    void _PrepareTexture()  override;
    void _PrepareMaterial() override;
    void _PrepareShader()   override;

private:
    int _GetShaderMacros() const;

private:
    int     m_row       = 1;
    int     m_col       = 1;
    float   m_spacing   = 2;
    float   m_radius    = 1;
    int     m_stacks    = 18;
    int     m_sectors   = 36;

    struct MaterialSource
    {
        int  NumOfLights                = 4;
        bool HasAlbedoMap               = false;
        bool HasAoMap                   = false;
        bool HasMetallicMap             = false;
        bool HasRoughnessMap            = false;
        bool HasDiffuseIrradianceMap    = false;
        bool HasSpecularIrradianceMap   = false;
        bool HasNormalMap               = false;
        std::shared_ptr<Texture> AlbedoMap     = nullptr;
        std::shared_ptr<Texture> AoMap         = nullptr;
        std::shared_ptr<Texture> MetallicMap   = nullptr;
        std::shared_ptr<Texture> RoughnessMap  = nullptr;
        std::shared_ptr<Texture> IrradianceMap = nullptr;
        std::shared_ptr<Texture> PrefilterMap  = nullptr;
        std::shared_ptr<Texture> LUTofBRDF     = nullptr;
        std::shared_ptr<Texture> NormalMap     = nullptr;
    }
    m_materialSource;
};
                                                  
class REContainers : public Renderer::Element
{
public:
    REContainers(const std::string& name="RE_Containers_unnamed") : Renderer::Element(name) {}
    static std::string GetTypeName() { return "RE_Containers"; }
    static std::shared_ptr<REContainers> Create(const std::string& name) { return std::make_shared<REContainers>(name); }
    std::shared_ptr<Renderer::Element> Set(unsigned int nInstances, float radius, float offset);

    bool OnImgui() override;

protected:
    void _PrepareMesh()     override;
    void _PrepareTexture()  override;
    void _PrepareMaterial() override;
    void _PrepareShader()   override;

private:
    int _GetShaderMacros() const;

private:
    float m_radius = 50.0f;
    float m_offset = 20.0f;
    struct 
    {
        glm::vec3* diffuseReflectance = nullptr;
        glm::vec3* specularReflectance = nullptr;
        glm::vec3* emissiveColor = nullptr;
        glm::vec3* ambientColor = nullptr;
        float* shininess = nullptr;
        float* displacementScale = nullptr;
        float* emissiveIntensity = nullptr;
        float* bloomThreshold = nullptr;
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
    m_materialSource; 
};

class REGroundPlane : public Renderer::Element
{
public:
    REGroundPlane(const std::string& name) : Renderer::Element(name) { _Prepare(); }
    static std::string GetTypeName() { return "RE_GroundPlane"; }
    static std::shared_ptr<REGroundPlane> Create(const std::string& name) { return std::make_shared<REGroundPlane>(name); }

    bool OnImgui() override;

protected:
    void _PrepareMesh()     override;
    void _PrepareTexture()  override;
    void _PrepareMaterial() override;
    void _PrepareShader()   override;
private:
};
