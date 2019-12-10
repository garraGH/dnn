
/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : sandbox/layer_learnopengl.cpp
* author      : Garra
* time        : 2019-10-27 11:06:13
* description : 
*
============================================*/


#include "layer_learnopengl.h"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/string_cast.hpp"
#include "osdialog.h"

std::shared_ptr<LearnOpenGLLayer> LearnOpenGLLayer::Create()
{
    return std::make_shared<LearnOpenGLLayer>();
}

LearnOpenGLLayer::LearnOpenGLLayer()
    : Layer( "LearnOpenGLLayer" )
{
    const std::shared_ptr<Camera>& cam = m_vpOffscreen->GetCamera();
    cam->SetFrameBuffer(m_fbOffscreenHDR);
    cam->SetTarget(glm::vec3(0, 8, 0));
    cam->SetPosition(glm::vec3(0, 1, 5));
    cam->SetTarget(glm::vec3(0));

    m_vpBase->AttachCamera(cam);
    m_vpBright->AttachCamera(cam);
    m_vpBlur->AttachCamera(cam);
    m_vpBloom->AttachCamera(cam);

//     m_vpBase->SetRange(0, 0.5, 0.5, 0.5);
//     m_vpBright->SetRange(0.5, 0.5, 0.5, 0.5);
//     m_vpBlur->SetRange(0, 0, 0.5, 0.5);
//     m_vpBloom->SetRange(0.5, 0, 0.5, 0.5);

//     m_fbMS->AddColorBuffer("BaseColorBuffer", Texture::Format::RGB16F);
//     m_fbMS->AddColorBuffer("BrightColorBuffer", Texture::Format::RGB16F);
//     m_fbMS->AddRenderBuffer("DepthStencil", RenderBuffer::Format::DEPTH24_STENCIL8);
//     m_fbSS->AddColorBuffer("BaseColorBuffer", Texture::Format::RGB16F);
//     m_fbBlurH->AddColorBuffer("HorizontalBlurColorBuffer", Texture::Format::RGB16F);
//     m_fbBlurV->AddColorBuffer("VerticalBlurColorBuffer", Texture::Format::RGB16F);

    _PrepareUniformBuffers();
    _PrepareSkybox();
    _PrepareOffscreenPlane();
    _PrepareUnitCubic();
    _PrepareSphere(1, 18, 36);
    _PrepareSpheresPBR(1, 18, 36);
    _PrepareGroundPlane();
//     _PrepareModel();
}

void LearnOpenGLLayer::OnEvent(Event& e)
{
    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(BIND_EVENT_CALLBACK(LearnOpenGLLayer, _On##event))
    DISPATCH(WindowResizeEvent);
#undef DISPATCH
    if(m_splitViewport)
    {
        m_vpBase->OnEvent(e);
        m_vpBright->OnEvent(e);
        m_vpBlur->OnEvent(e);
    }
    m_vpBloom->OnEvent(e);
    m_vpOffscreen->OnEvent(e);
}

bool LearnOpenGLLayer::_OnWindowResizeEvent(WindowResizeEvent& e)
{
    INFO("LearnOpenGLLayer::_OnWindowResizeEvent");
    float w = e.GetWidth();
    float h = e.GetHeight();

    if(m_splitViewport)
    {
        w *= 0.5;
        h *= 0.5;
    }

    m_rightTopTexCoord->x = w/m_offscreenBufferSize.x;
    m_rightTopTexCoord->y = h/m_offscreenBufferSize.y;

    return false;
}

void LearnOpenGLLayer::_UpdateShaderID()
{
    m_shaderID = 0;
    if(m_material.hasDiffuseReflectance)
    {
        m_shaderID |= static_cast<int>(Shader::Macro::DIFFUSE_REFLECTANCE);
    }
    if(m_material.hasSpecularReflectance)
    {
        m_shaderID |= static_cast<int>(Shader::Macro::SPECULAR_REFLECTANCE);
    }
    if(m_material.hasEmissiveColor)
    {
        m_shaderID |= static_cast<int>(Shader::Macro::EMISSIVE_COLOR);
    }
    if(m_material.hasDiffuseMap)
    {
        m_shaderID |= static_cast<int>(Shader::Macro::DIFFUSE_MAP);
    }
    if(m_material.hasSpecularMap)
    {
        m_shaderID |= static_cast<int>(Shader::Macro::SPECULAR_MAP);
    }
    if(m_material.hasEmissiveMap)
    {
        m_shaderID |= static_cast<int>(Shader::Macro::EMISSIVE_MAP);
    }
    if(m_material.hasNormalMap)
    {
        m_shaderID |= static_cast<int>(Shader::Macro::NORMAL_MAP);
    }
    if(m_material.hasDisplacementMap)
    {
        m_shaderID |= static_cast<int>(Shader::Macro::DISPLACEMENT_MAP);
    }
}

void LearnOpenGLLayer::_UpdateShaderID_HDR()
{
    m_shaderID_HDR = 0;
    if(m_material_HDR.enableToneMap)
    {
        m_shaderID_HDR |= static_cast<int>(Shader::Macro::TONE_MAP);
    }
    if(m_material_HDR.enableGammaCorrection)
    {
        m_shaderID_HDR |= static_cast<int>(Shader::Macro::GAMMA_CORRECTION);
    }
}

std::string LearnOpenGLLayer::_StringOfShaderID() const
{
    return std::to_string(m_shaderID);
}

std::string LearnOpenGLLayer::_StringOfShaderID_HDR()  const
{
    return std::to_string(m_shaderID_HDR);
}

void LearnOpenGLLayer::OnUpdate(float deltaTime)
{
    _UpdateMaterialUniforms();

    _RenderToTexture_HDR();
    _RenderToScreen_HDR();
    _RenderToTexture_Blur();
    _RenderToScreen_Blur();
    _RenderToTexture_Bloom();
    _RenderToScreen_Bloom();

    m_vpOffscreen->OnUpdate(deltaTime);
}

void LearnOpenGLLayer::_RenderToTexture_HDR()
{
//     Renderer::SetPolygonMode(Renderer::PolygonMode::LINE);
    Renderer::BeginScene(m_vpOffscreen, m_fbOffscreenHDR);
    if(m_showSky)
    {
        Renderer::Submit("Skybox", "Skybox");
    }
    if(m_showGround)
    {
        Renderer::Submit("GroundPlane", "GroundPlane");
    }
//     m_planet->Draw(m_shaderColor);
//     m_rock->Draw(m_shaderColor);
//     Renderer::Submit("UnitCubic", "Blinn-Phong-Instance", m_numOfInstance);
//     Renderer::Submit(m_eleCubic, m_shaderOfMaterial);
//     Renderer::Submit(m_eleSphere, m_shaderSphere, m_numOfLights);
    Renderer::Submit(m_eleSpherePBR0, m_shaderSpherePBR0, m_row*m_col);
    Renderer::EndScene();
}

void LearnOpenGLLayer::_RenderToScreen_HDR()
{
//     Renderer::SetPolygonMode(Renderer::PolygonMode::FILL);
    if(!m_splitViewport)
    {
        return;
    }

    Renderer::BeginScene(m_vpBase);
    Renderer::Submit(m_eleBase, m_shaderHDR);
    Renderer::EndScene();

    Renderer::BeginScene(m_vpBright);
    Renderer::Submit(m_eleBright, m_shaderHDR);
    Renderer::EndScene();
}

void LearnOpenGLLayer::_RenderToTexture_Blur()
{
    for(int i=0; i<m_blurIteration; i++)
    {
        Renderer::BeginScene(m_vpOffscreen, m_fbOffscreenBlurPing);
        Renderer::Submit(m_eleBlurH, m_shaderBlur);
        Renderer::EndScene();
        Renderer::BeginScene(m_vpOffscreen, m_fbOffscreenBlurPong);
        Renderer::Submit(m_eleBlurV, m_shaderBlur);
        Renderer::EndScene();
    }
}

void LearnOpenGLLayer::_RenderToScreen_Blur()
{
    if(!m_splitViewport)
    {
        return;
    }

    Renderer::BeginScene(m_vpBlur);
    Renderer::Submit(m_eleBlurH, m_shaderHDR);
    Renderer::EndScene();
}


void LearnOpenGLLayer::_RenderToTexture_Bloom()
{
    Renderer::BeginScene(m_vpOffscreen, m_fbOffscreenBloom);
    Renderer::Submit(m_eleBloom, m_shaderBloom);
    Renderer::EndScene();
}

void LearnOpenGLLayer::_RenderToScreen_Bloom()
{
    Renderer::BeginScene(m_vpBloom);
    Renderer::Submit(m_eleBloom, m_shaderHDR);
    Renderer::EndScene();
}

/*
void LearnOpenGLLayer::OnUpdate(float deltaTime)
{
    _UpdateMaterialUniforms();

    Renderer::BeginScene(m_vpBase, m_fbMS);
//     Renderer::BeginScene(m_vpBase);
//     m_crysisNanoSuit->Draw(m_shaderPos);
//     m_crysisNanoSuit->Draw(m_shaderBlinnPhong);
//     m_trailer->Draw(m_shaderBlinnPhong);
//     m_silkingMachine->Draw(m_shaderBlinnPhong);
//     m_horse->Draw(m_shaderBlinnPhong);
//     m_bulb->Draw(m_shaderColor);
//     m_handLight->Draw(m_shaderColor);
    if(m_showGround)
        Renderer::Submit("GroundPlane", "GroundPlane");
    if(m_showSky)
        Renderer::Submit("Skybox", "Skybox");
    Renderer::Submit("UnitCubic", "Blinn-Phong-Instance", m_numOfInstance);
    Renderer::Submit(m_eleCubic, m_shaderOfMaterial);
    Renderer::EndScene();                       

//     Renderer::BlitFrameBuffer(m_fbMS, m_fbSS);

    Renderer::BeginScene(m_vpBase);
    Renderer::Submit(m_eleBase, m_shaderHDR);
    Renderer::EndScene();

    Renderer::BeginScene(m_vpBright);
    Renderer::Submit(m_eleBright, m_shaderHDR);
    Renderer::EndScene();
 

//     Renderer::Resources::Get<Material>("BlurH")->SetTexture("u_Offscreen", m_fbMS->GetColorBuffer("BrightColorBuffer"));
    Renderer::BeginScene(m_vpBase, m_fbBlurH);
    Renderer::Submit(m_eleBlurH, m_shaderBlur);
    Renderer::EndScene();
    Renderer::Resources::Get<Material>("BlurH")->SetTexture("u_Offscreen", m_fbBlurV->GetColorBuffer("VerticalBlurColorBuffer"));

    for(int i=0; i<m_blurIteration; i++)
    {
        Renderer::BeginScene(m_vpBase, m_fbBlurV);
        Renderer::Submit(m_eleBlurV, m_shaderBlur);
        Renderer::EndScene();

        Renderer::BeginScene(m_vpBase, m_fbBlurH);
        Renderer::Submit(m_eleBlurH, m_shaderBlur);
        Renderer::EndScene();
    }

    Renderer::BeginScene(m_vpBlur);
    Renderer::Submit(m_eleBlurV, m_shaderHDR);
    Renderer::EndScene();

//     Renderer::BeginScene(m_vpBloom);
//     Renderer::Submit(m_eleBloom, m_shaderBloom);
//     Renderer::EndScene();

    m_vpBase->OnUpdate(deltaTime);
}
*/

void LearnOpenGLLayer::OnImGuiRender()
{
    using MU = Material::Uniform;
    m_vpOffscreen->OnImGuiRender();

    ImGui::Begin("LearnOpenGLLayer");




    ImGui::Checkbox("ShowSky", &m_showSky);
    ImGui::Checkbox("ShowGround", &m_showGround);
    if(ImGui::Checkbox("SplitViewport", &m_splitViewport))
    {
        if(m_splitViewport)
        {
            m_vpBloom->SetRange(0.5, 0, 0.5, 0.5);
            m_vpOffscreen->SetRange(0, 0, 0.5, 0.5);
            m_rightTopTexCoord->x *= 0.5;
            m_rightTopTexCoord->y *= 0.5;
        }
        else
        {
            m_vpBloom->SetRange(0, 0, 1, 1);
            m_vpOffscreen->SetRange(0, 0, 1, 1);
            m_rightTopTexCoord->x *= 2;
            m_rightTopTexCoord->y *= 2;
        }
    }

    ImGui::Separator();

    ImGui::DragInt("BlurIteration", &m_blurIteration, 1, 0, 20);
    ImGui::DragFloat("BloomThreshold", m_bloomThreshold, 0.1f, 0, 10);

//     if(ImGui::InputInt("Samples", (int*)&m_samples))
//     {
//         unsigned int w = m_fbMS->GetWidth();
//         unsigned int h = m_fbMS->GetHeight();
//         m_fbMS->Reset(w, h, m_samples);
//     }
    if(ImGui::CollapsingHeader("PostProcess"))
    {
#define RadioButton(x)                                                      \
        if(ImGui::RadioButton(#x, m_pp == PostProcess::x))                  \
        {                                                                   \
            m_pp = PostProcess::x;                                          \
            Renderer::Resources::Get<MU>("PostProcess")->UpdateData(&m_pp); \
        }                                                                   \

        RadioButton(None);
        RadioButton(Gray);
        RadioButton(Smooth);
        RadioButton(Edge);
#undef RadioButton
    }

    ImGui::Separator();
    ImGui::PushItemWidth(200);
    if(ImGui::CollapsingHeader("Environment"))
    {
        ImGui::ColorPicker3("AmbientColor", reinterpret_cast<float*>(m_ambientColor));
    }

    if(ImGui::CollapsingHeader("Material"))
    {
        ImGui::Indent();
        bool bShaderChanged = false;
        bShaderChanged |= ImGui::Checkbox("DiffuseRelectance", &m_material.hasDiffuseReflectance);
        ImGui::SameLine(200);
        ImGui::ColorEdit3("DiffuseRelectance", (float*)m_material.diffuseReflectance, ImGuiColorEditFlags_NoInputs|ImGuiColorEditFlags_NoLabel);

        bShaderChanged |= ImGui::Checkbox("SpecularReflectance", &m_material.hasSpecularReflectance);
        ImGui::SameLine(200);
        ImGui::ColorEdit3("SpecularReflectance", (float*)m_material.specularReflectance, ImGuiColorEditFlags_NoInputs|ImGuiColorEditFlags_NoLabel);
        ImGui::SameLine(250);
        ImGui::SetNextItemWidth(64);
        ImGui::DragFloat("Shininess", m_material.shininess, 2, 2, 512, "%.0f");

        bShaderChanged |= ImGui::Checkbox("EmissiveColor", &m_material.hasEmissiveColor);
        ImGui::SameLine(200);
        ImGui::ColorEdit3("EmissiveColor", (float*)m_material.emissiveColor, ImGuiColorEditFlags_NoInputs|ImGuiColorEditFlags_NoLabel);
        ImGui::SameLine(250);
        ImGui::SetNextItemWidth(64);
        ImGui::DragFloat("Intensity", m_material.emissiveIntensity, 0.1, 0, 10);

        bShaderChanged |= ImGui::Checkbox("DiffuseMap", &m_material.hasDiffuseMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_material.diffuseMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_material.diffuseMap);
        }
        bShaderChanged |= ImGui::Checkbox("SpecularMap", &m_material.hasSpecularMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_material.specularMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_material.specularMap);
        }
        bShaderChanged |= ImGui::Checkbox("EmissiveMap", &m_material.hasEmissiveMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_material.emissiveMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_material.emissiveMap);
        }
        bShaderChanged |= ImGui::Checkbox("NormalMap", &m_material.hasNormalMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_material.normalMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_material.normalMap);
        }
        bShaderChanged |= ImGui::Checkbox("DisplacementMap", &m_material.hasDisplacementMap);
        ImGui::SameLine(200);
        if(ImGui::ImageButton((void*)(intptr_t)m_material.displacementMap->ID(), ImVec2(16, 16), ImVec2(0, 0), ImVec2(1, 1), 1, ImColor(0, 128, 0, 128)))
        {
            _UpdateTexture(m_material.displacementMap);
        }
        ImGui::SameLine(250);
        ImGui::SetNextItemWidth(64);
        ImGui::DragFloat("DisplacementScale", m_material.displacementScale,  0.001,  0,  1);

        if(bShaderChanged)
        {
            _UpdateShaderID();
            std::string shaderName = _StringOfShaderID();
            if(Renderer::Resources::Exist<Shader>(shaderName))
            {
                m_shaderOfMaterial = Renderer::Resources::Get<Shader>(shaderName);
            }
            else
            {
                m_shaderOfMaterial = Renderer::Resources::Create<Shader>(shaderName)->Define(m_shaderID)->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");
            }
        }
        ImGui::Unindent();
    }

    if(ImGui::CollapsingHeader("Material_HDR"))
    {
        bool bChanged = false;
        ImGui::Indent();
        bChanged |= ImGui::Checkbox("ToneMap", &m_material_HDR.enableToneMap);
        bChanged |= ImGui::Checkbox("GammaCorrection", &m_material_HDR.enableGammaCorrection);
        if(bChanged)
        {
            _UpdateShaderID_HDR();
            std::string shaderName = _StringOfShaderID_HDR();
            if(Renderer::Resources::Exist<Shader>(shaderName))
            {
                m_shaderHDR = Renderer::Resources::Get<Shader>(shaderName);
            }
            else
            {
                m_shaderHDR = Renderer::Resources::Create<Shader>(shaderName)->Define(m_shaderID_HDR)->LoadFromFile("/home/garra/study/dnn/assets/shader/OffscreenTexture.glsl");
            }
        }
        
        ImGui::DragFloat("Gamma", m_material_HDR.gamma,  0.01,  0.1,  3.0);
        ImGui::DragFloat("Exposure", m_material_HDR.exposure,  0.01,  0,  10);
        ImGui::Unindent();
    }
    if(ImGui::CollapsingHeader("Light"))
    {
        ImGui::Indent();
        ImGui::SetNextWindowPos({200, 0});
        if(ImGui::CollapsingHeader("DirectionalLight"))
        {
            bool bChanged = ImGui::ColorPicker3("Color", glm::value_ptr(m_dLight.clr));
            bChanged |= ImGui::DragFloat3("Direction", glm::value_ptr(m_dLight.dir), 0.1f, -1, 1);
            bChanged |= ImGui::DragFloat("Intensity0", &m_dLight.intensity,  0.1,  0,  10);
            if(bChanged)
            {
                Renderer::Resources::Get<UniformBuffer>("Light")->Upload("DirectionalLight", &m_dLight);
            }
        }
        if(ImGui::CollapsingHeader("PointLight"))
        {
            bool bChanged = ImGui::ColorPicker3("Color", glm::value_ptr(m_pLight.clr));
            bChanged |= ImGui::DragFloat3("Position", glm::value_ptr(m_pLight.pos),  0.1f,  -10.0,  10.0);
            bChanged |= ImGui::InputFloat3("AttenuationCoefficents", glm::value_ptr(m_pLight.coe));
            bChanged |= ImGui::DragFloat("Intensity1", &m_pLight.intensity,  0.1,  0,  10);
            if(bChanged)
            {
                Renderer::Resources::Get<UniformBuffer>("Light")->Upload("PointLight", &m_pLight);
            }
        }
        if(ImGui::CollapsingHeader("SpotLight"))
        {
            bool bChanged = ImGui::ColorPicker3("Color", glm::value_ptr(m_sLight.clr));
            bChanged |= ImGui::DragFloat3("Position", glm::value_ptr(m_sLight.pos), 0.1f, -10, 10);
            bChanged |= ImGui::DragFloat3("Direction", glm::value_ptr(m_sLight.dir), 0.1f, -1, 1);
            bChanged |= ImGui::InputFloat3("AttenuationCoefficents", glm::value_ptr(m_sLight.coe));
            bChanged |= ImGui::DragFloat("Intensity2", &m_sLight.intensity,  0.1,  0,  10);

            if(ImGui::DragFloat("InnerCone", &m_sLight.degInnerCone, 1, 0, 60))
            {
                bChanged = true;
                m_sLight.cosInnerCone = std::cos(glm::radians(m_sLight.degInnerCone));
                if(m_sLight.degOuterCone<m_sLight.degInnerCone)
                {
                    m_sLight.degOuterCone = m_sLight.degInnerCone;
                    m_sLight.cosOuterCone = m_sLight.cosInnerCone;
                }
            }
            if(ImGui::DragFloat("OuterCone", &m_sLight.degOuterCone, 1, 0, 90))
            {
                bChanged = true;
                m_sLight.cosOuterCone = std::cos(glm::radians(m_sLight.degOuterCone));
                if(m_sLight.degInnerCone>m_sLight.degOuterCone)
                {
                    m_sLight.degInnerCone = m_sLight.degOuterCone;
                    m_sLight.cosInnerCone = m_sLight.cosOuterCone;
                }
            }

            if(bChanged)
            {
                Renderer::Resources::Get<UniformBuffer>("Light")->Upload("SpotLight", &m_sLight);
            }
        }
        if(ImGui::CollapsingHeader("FlashLight"))
        {
            bool bChanged = ImGui::ColorPicker3("Color", glm::value_ptr(m_fLight.clr));
            bChanged |= ImGui::InputFloat3("AttenuationCoefficents", glm::value_ptr(m_fLight.coe));
            bChanged |= ImGui::DragFloat("Intensity3", &m_fLight.intensity,  0.1,  0,  10);
            ImGui::LabelText("Position", "%.1f, %.1f, %.1f", m_fLight.pos.x, m_fLight.pos.y, m_fLight.pos.z);
            ImGui::LabelText("Direction", "%.1f, %.1f, %.1f", m_fLight.dir.x, m_fLight.dir.y, m_fLight.dir.z);
            if(ImGui::DragFloat("InnerCone", &m_fLight.degInnerCone, 1, 0, 30))
            {
                bChanged = true;
                m_fLight.cosInnerCone = std::cos(glm::radians(m_fLight.degInnerCone));
                if(m_fLight.degOuterCone<m_fLight.degInnerCone)
                {
                    m_fLight.degOuterCone = m_fLight.degInnerCone;
                    m_fLight.cosOuterCone = m_fLight.cosInnerCone;
                }
            }
            if(ImGui::DragFloat("OuterCone", &m_fLight.degOuterCone, 1, 0, 60))
            {
                bChanged = true;
                m_fLight.cosOuterCone = std::cos(glm::radians(m_fLight.degOuterCone));
                if(m_fLight.degInnerCone>m_fLight.degOuterCone)
                {
                    m_fLight.degInnerCone = m_fLight.degOuterCone;
                    m_fLight.cosInnerCone = m_fLight.cosOuterCone;
                }
            }
            if(bChanged)
            {
                Renderer::Resources::Get<UniformBuffer>("Light")->Upload("FlashLight", &m_fLight);
            }
        }
        ImGui::Unindent();
    }

    ImGui::End();
}

void LearnOpenGLLayer::_UpdateTexture(std::shared_ptr<Texture>& tex)
{
    char* filename = osdialog_file(OSDIALOG_OPEN, "/home/garra/study/dnn/assets/texture", nullptr, nullptr);
    if(filename)
    {
        tex->Reload(filename);
        delete[] filename;
    }
}
void LearnOpenGLLayer::_PrepareModel()
{
    m_planet = Renderer::Resources::Create<Model>("Planet")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Planet/planet.obj");
    m_rock = Renderer::Resources::Create<Model>("Rock")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Rock/rock.obj");
//     m_crysisNanoSuit = Renderer::Resources::Create<Model>("CysisNanoSuit")->LoadFromFile("/home/garra/study/dnn/assets/mesh/CysisNanoSuit/scene.fbx");
//     m_silkingMachine = Renderer::Resources::Create<Model>("SilkingMachine")->LoadFromFile("/home/garra/study/dnn/assets/mesh/SilkingMachine/SilkingMachine.fbx");
//     m_horse = Renderer::Resources::Create<Model>("Horse")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Horse/Horse.fbx");
//     m_trailer = Renderer::Resources::Create<Model>("Trailer")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Trailer/Alena_Shek.obj");
//     m_bulb = Renderer::Resources::Create<Model>("Bulb")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Bulb/Bulbs.3ds");
//     m_handLight = Renderer::Resources::Create<Model>("HandLight")->LoadFromFile("/home/garra/study/dnn/assets/mesh/HandLight/hand_light.blend");
//     m_shaderPos = Renderer::Resources::Create<Shader>("Pos")->LoadFromFile("/home/garra/study/dnn/assets/shader/Model.glsl");
//     Renderer::Resources::Create<Shader>("Default")->LoadFromFile("/home/garra/study/dnn/assets/shader/Default.glsl");
//     STOP
}

void LearnOpenGLLayer::_PrepareUnitCubic()
{
    float vertices[] =   // (a_PositionMS, a_NormalMS, a_TangentMS, a_TexCoord)
    {
        // front 
        -1, -1, +1,  0, 0, +1,  +1, 0, 0,  0, 0,   
        +1, -1, +1,  0, 0, +1,  +1, 0, 0,  1, 0, 
        +1, +1, +1,  0, 0, +1,  +1, 0, 0,  1, 1, 
        -1, +1, +1,  0, 0, +1,  +1, 0, 0,  0, 1, 
        // back                            
        +1, -1, -1,  0, 0, -1,  -1, 0, 0,  0, 0, 
        -1, -1, -1,  0, 0, -1,  -1, 0, 0,  1, 0, 
        -1, +1, -1,  0, 0, -1,  -1, 0, 0,  1, 1, 
        +1, +1, -1,  0, 0, -1,  -1, 0, 0,  0, 1, 
        // left                            
        -1, -1, -1,  -1, 0, 0,  0, 0, +1,  0, 0, 
        -1, -1, +1,  -1, 0, 0,  0, 0, +1,  1, 0, 
        -1, +1, +1,  -1, 0, 0,  0, 0, +1,  1, 1, 
        -1, +1, -1,  -1, 0, 0,  0, 0, +1,  0, 1, 
        // right                           
        +1, -1, +1,  +1, 0, 0,  0, 0, -1,  0, 0, 
        +1, -1, -1,  +1, 0, 0,  0, 0, -1,  1, 0, 
        +1, +1, -1,  +1, 0, 0,  0, 0, -1,  1, 1, 
        +1, +1, +1,  +1, 0, 0,  0, 0, -1,  0, 1, 
        // up                              
        -1, +1, +1,  0, +1, 0,  +1, 0, 0,  0, 0, 
        +1, +1, +1,  0, +1, 0,  +1, 0, 0,  1, 0, 
        +1, +1, -1,  0, +1, 0,  +1, 0, 0,  1, 1, 
        -1, +1, -1,  0, +1, 0,  +1, 0, 0,  0, 1, 
        // down                            
        -1, -1, -1,  0, -1, 0,  -1, 0, 0,  0, 0, 
        +1, -1, -1,  0, -1, 0,  -1, 0, 0,  1, 0, 
        +1, -1, +1,  0, -1, 0,  -1, 0, 0,  1, 1, 
        -1, -1, +1,  0, -1, 0,  -1, 0, 0,  0, 1, 
    };

    unsigned char indices[] = 
    { 
        0, 1, 2, 
        0, 2, 3, 
        4, 5, 6, 
        4, 6, 7, 
        8, 9, 10, 
        8, 10, 11, 
        12, 13, 14, 
        12, 14, 15, 
        16, 17, 18, 
        16, 18, 19, 
        20, 21, 22, 
        20, 22, 23
    };
    glm::mat4 matM2W[m_numOfInstance];
    srand(time(NULL));
    float radius = 50.0f;
    float offset = 20.0f;

    int k = 0;
    std::shared_ptr<Transform> tf = Transform::Create("temp");
    glm::vec3 translation = glm::vec3(0);
    glm::vec3 rotation = glm::vec3(0);
    glm::vec3 scale = glm::vec3(1);
    for(unsigned int i=0; i<m_numOfInstance; i++)
    {
        float angle = i*360.0f/m_numOfInstance;
        translation.x = std::sin(angle)*radius+((rand()%(int)(2*offset*100))/100.0f-offset);
        translation.y = 0.4f*((rand()%(int)(2*offset*100))/100.0f-offset);
        translation.z = std::cos(angle)*radius+((rand()%(int)(2*offset*100))/100.0f-offset);

        rotation.x = rand()%360;
        rotation.y = rand()%360;
        rotation.z = rand()%360;

        scale.x = (rand()%20)/20.0f+0.05f;
        scale.z = scale.y = scale.x;
        
        matM2W[k++] = tf->Set(translation, rotation, scale)->Get();
    }

    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float3, "a_PositionMS", false}, 
                                    {Buffer::Element::DataType::Float3, "a_NormalMS", false},
                                    {Buffer::Element::DataType::Float3, "a_TangentMS", false},
                                    {Buffer::Element::DataType::Float2, "a_TexCoord", false}  };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    Buffer::Layout layoutInstance = { {Buffer::Element::DataType::Mat4, "a_MS2WS", false, 1} };

    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> indexBuffer = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Buffer> instanceBuffer = Buffer::CreateVertex(m_numOfInstance*sizeof(glm::mat4), matM2W)->SetLayout(layoutInstance);
    std::shared_ptr<Elsa::Mesh> mesh= Renderer::Resources::Create<Elsa::Mesh>("UnitCubic")->Set(indexBuffer, {vertexBuffer, instanceBuffer});
    using MU = Material::Uniform;


    std::shared_ptr<MU> maMaterialDiffuseReflectance = Renderer::Resources::Create<MU>("MaterialDiffuseReflectance")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.8f)));
    std::shared_ptr<MU> maMaterialSpecularReflectance = Renderer::Resources::Create<MU>("MaterialSpecularReflectance")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0f)));
    std::shared_ptr<MU> maMaterialEmissiveColor = Renderer::Resources::Create<MU>("MaterialEmissiveColor")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1f)));
    std::shared_ptr<MU> maMaterialEmissiveIntensity = Renderer::Resources::Create<MU>("MaterialEmissiveIntensity")->Set(MU::Type::Float1);
    std::shared_ptr<MU> maMaterialShininess = Renderer::Resources::Create<MU>("MaterialShininess")->SetType(MU::Type::Float1);
    std::shared_ptr<MU> maMaterialDisplacementScale = Renderer::Resources::Create<MU>("MaterialDisplacementScale")->SetType(MU::Type::Float1);
    std::shared_ptr<MU> maCameraPosition = Renderer::Resources::Create<MU>("CameraPosition")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(2.0f)));
    std::shared_ptr<MU> maBloomThreshold = Renderer::Resources::Create<MU>("BloomThreshold")->SetType(MU::Type::Float1);


    m_material.diffuseReflectance = reinterpret_cast<glm::vec3*>(maMaterialDiffuseReflectance->GetData());
    m_material.specularReflectance = reinterpret_cast<glm::vec3*>(maMaterialSpecularReflectance->GetData());
    m_material.emissiveColor = reinterpret_cast<glm::vec3*>(maMaterialEmissiveColor->GetData());
    m_material.emissiveIntensity = reinterpret_cast<float*>(maMaterialEmissiveIntensity->GetData());
    *m_material.emissiveIntensity = 1.0f;
    m_material.shininess = reinterpret_cast<float*>(maMaterialShininess->GetData());
    *m_material.shininess = 32.0f;
    m_material.displacementScale = reinterpret_cast<float*>(maMaterialDisplacementScale->GetData());
    *m_material.displacementScale = 0.1f;
    m_material.diffuseMap = Renderer::Resources::Create<Texture2D>("DiffuseMap")->Load("/home/garra/study/dnn/assets/texture/wood.png");
    m_material.normalMap = Renderer::Resources::Create<Texture2D>("NormalMap")->Load("/home/garra/study/dnn/assets/texture/toy_box_normal.png");
    m_material.displacementMap = Renderer::Resources::Create<Texture2D>("DisplacementMap")->Load("/home/garra/study/dnn/assets/texture/toy_box_disp.png");
//     m_material.diffuseMap = Renderer::Resources::Create<Texture2D>("DiffuseMap")->Load("/home/garra/study/dnn/assets/texture/bricks2.jpg");
//     m_material.normalMap = Renderer::Resources::Create<Texture2D>("NormalMap")->Load("/home/garra/study/dnn/assets/texture/bricks2_normal.jpg");
//     m_material.depthMap = Renderer::Resources::Create<Texture2D>("DepthMap")->Load("/home/garra/study/dnn/assets/texture/bricks2_disp.jpg");
//     m_material.diffuseMap = Renderer::Resources::Create<Texture2D>("DiffuseMap")->Load("/home/garra/study/dnn/assets/texture/container2.png");
    m_material.specularMap = Renderer::Resources::Create<Texture2D>("SpecularMap")->Load("/home/garra/study/dnn/assets/texture/lighting_maps_specular_color.png");
    m_material.emissiveMap = Renderer::Resources::Create<Texture2D>("EmissiveMap")->Load("/home/garra/study/dnn/assets/texture/matrix.jpg");


    // AmbientColor
    std::shared_ptr<MU> maAmbientColor = Renderer::Resources::Create<MU>("AmbientColor")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.3f)));
    m_ambientColor = reinterpret_cast<glm::vec3*>(maAmbientColor->GetData());

    m_bloomThreshold = reinterpret_cast<float*>(maBloomThreshold->GetData());
    *m_bloomThreshold = 1.0f;

    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("UnitCubic");
    mtr->SetUniform("u_Material.DiffuseReflectance", maMaterialDiffuseReflectance);
    mtr->SetUniform("u_Material.SpecularReflectance", maMaterialSpecularReflectance);
    mtr->SetUniform("u_Material.EmissiveColor", maMaterialEmissiveColor);
    mtr->SetUniform("u_Material.EmissiveIntensity", maMaterialEmissiveIntensity);
    mtr->SetUniform("u_Material.Shininess", maMaterialShininess);
    mtr->SetUniform("u_Material.DisplacementScale", maMaterialDisplacementScale);
    mtr->SetTexture("u_Material.DiffuseMap", m_material.diffuseMap);
    mtr->SetTexture("u_Material.NormalMap", m_material.normalMap);
    mtr->SetTexture("u_Material.SpecularMap", m_material.specularMap);
    mtr->SetTexture("u_Material.EmissiveMap", m_material.emissiveMap);
    mtr->SetTexture("u_Material.DisplacementMap", m_material.displacementMap);

    mtr->SetUniform("u_Camera.PositionWS", maCameraPosition);
    // AmbientColor
    mtr->SetUniform("u_AmbientColor", maAmbientColor);
    mtr->SetUniform("u_BloomThreshold", maBloomThreshold);

    mtr->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("Transform"));
    mtr->SetUniformBuffer("Light", Renderer::Resources::Get<UniformBuffer>("Light"));

    Renderer::Resources::Create<Shader>("Blinn-Phong-Instance")->Define("INSTANCE|DIFFUSE_MAP|NORMAL_MAP")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");
    m_shaderBlinnPhong = Renderer::Resources::Create<Shader>("Blinn-Phong")->Define("DIFFUSE_MAP|NORMAL_MAP")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");
    m_shaderOfMaterial = Renderer::Resources::Create<Shader>(_StringOfShaderID())->Define(m_shaderID)->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");

    m_eleCubic = Renderer::Resources::Create<Renderer::Element>("UnitCubic")->Set(mesh, mtr);


//     unsigned int id = m_shaderBlinnPhong->ID();
//     GLuint index = glGetUniformBlockIndex(id, "Light");
//     INFO("Light: index {}", index);
//     if(GL_INVALID_INDEX != index)
//     {
//         GLint size = 0;
//         glGetActiveUniformBlockiv(id, index, GL_UNIFORM_BLOCK_DATA_SIZE, &size);
//         INFO("BlockDataSize: {}", size);
//     }
// 
//     INFO("sizeof(SpotLight): {}", sizeof(m_sLight));
//     STOP
}

std::pair<std::vector<glm::vec3>, std::vector<glm::i16vec3>> LearnOpenGLLayer::_GenSphere(float radius, int stacks, int sectors)
{
    std::vector<glm::vec3> vertices;
    std::vector<glm::i16vec3> triangles;
    const float PI = 3.14159265;
    float x, y, z, xy;
    float stepStack = PI/stacks;
    float stepSector = 2*PI/sectors;
    float angleStack, angleSector;
    for(int i=0; i<=stacks; i++)
    {
        angleStack = PI/2-i*stepStack;
        xy = radius*cosf(angleStack);
        z = radius*sinf(angleStack);

        for(int j=0; j<=sectors; j++)
        {
            angleSector = j*stepSector;
            x = xy*cosf(angleSector);
            y = xy*sinf(angleSector);
            vertices.push_back({x, y, z});
        }
    }

    int k1, k2;
    for(int i=0; i<stacks; i++)
    {
        k1 = i*(sectors+1);
        k2 = k1+sectors+1;
        for(int j=0; j<sectors; j++, k1++, k2++)
        {
            if(i != 0)
            {
                triangles.push_back({k1, k2, k1+1});
            }
            if(i != stacks-1)
            {
                triangles.push_back({k1+1, k2, k2+1});
            }
        }
    }

    return { vertices, triangles };
}

void LearnOpenGLLayer::_PrepareSphere(float radius, int stacks, int sectors)
{
    auto [vertices, triangles] = _GenSphere(radius, stacks, sectors);

    std::shared_ptr<Transform> tf = Transform::Create("temp");
    glm::vec3 translation = glm::vec3(0);
    glm::vec3 rotation = glm::vec3(0);
    glm::vec3 scale = glm::vec3(1);
    int k = 0;
    srand(time(NULL));
    float offset = 20.0f;
    struct InstanceAttribute
    {
        glm::mat4 m2w;
        glm::vec3 color;
        float intensity;
    } instances[m_numOfLights];

    for(unsigned int i=0; i<m_numOfLights; i++, k++)
    {
        float angle = i*360.0f/m_numOfLights;
        translation.x = sinf(angle)*(rand()%100);
        translation.y = 0.4f*((rand()%(int)(2*offset*100))/100.0f-offset);
        translation.z = cosf(angle)*(rand()%100);

        scale.x = (rand()%100)/100.0f+0.01f;
        scale.z = scale.y = scale.x;
        
        instances[k].m2w = tf->Set(translation, rotation, scale)->Get();
        instances[k].color = { rand()%256/255.0f, rand()%256/255.0f, rand()%256/255.0f };
        instances[k].intensity = rand()%100/30.0f;
    }


    Buffer::Layout layoutVertex = { {Buffer::Element::DataType::Float3, "a_PositionMS", false} };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UShort} };
    Buffer::Layout layoutInstance = { {Buffer::Element::DataType::Mat4, "a_MS2WS", false, 1}, 
                                      {Buffer::Element::DataType::Float3, "a_Color", false, 1}, 
                                      {Buffer::Element::DataType::Float, "a_Intensity", false, 1} };
    
    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(vertices.size()*sizeof(glm::vec3), &vertices[0])->SetLayout(layoutVertex);
    std::shared_ptr<Buffer> indexBuffer = Buffer::CreateIndex(triangles.size()*sizeof(glm::i16vec3), &triangles[0])->SetLayout(layoutIndex);
    std::shared_ptr<Buffer> instanceBuffer = Buffer::CreateVertex(m_numOfLights*sizeof(InstanceAttribute), instances)->SetLayout(layoutInstance);
    std::shared_ptr<Elsa::Mesh> meshSphere = Renderer::Resources::Create<Elsa::Mesh>("Sphere")->Set(indexBuffer, {vertexBuffer, instanceBuffer});

    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("Sphere");
    mtr->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("Transform"));
    mtr->SetUniform("u_BloomThreshold", Renderer::Resources::Get<Material::Uniform>("BloomThreshold"));
    m_eleSphere = Renderer::Resources::Create<Renderer::Element>("Sphere")->Set(meshSphere, mtr);
    m_shaderSphere = Renderer::Resources::Create<Shader>("Sphere")->Define("INSTANCE")->LoadFromFile("/home/garra/study/dnn/assets/shader/Sphere.glsl");
}

void LearnOpenGLLayer::_PrepareSpheresPBR(float radius, int stacks, int sectors)
{
    int nInstance = m_row*m_col;
    struct VertexAttribute
    {
        glm::vec3 pos;
        glm::vec3 nor;
        glm::vec2 uv;
    };

    std::vector<VertexAttribute> vertices;
    std::vector<glm::i16vec3> triangles;
    const float PI = 3.14159265;
    float x, y, z, xz;
    float u, v;
    for(int i=0; i<=stacks; i++)
    {
        v = i/float(stacks);
        y = cosf(v*PI);
        xz = sinf(v*PI);
        for(int j=0; j<=sectors; j++)
        {
            u = j/float(sectors);
            x = xz*cosf(u*2*PI);
            z = xz*sinf(u*2*PI);
            vertices.push_back({ {radius*x, radius*y, radius*z}, {x, y, z}, {u, v} });
        }
    }

    int k1, k2;
    for(int i=0; i<stacks; i++)
    {
        k1 = i*(sectors+1);
        k2 = k1+sectors+1;
        for(int j=0; j<sectors; j++, k1++, k2++)
        {
            if(i != 0)
            {
                triangles.push_back({k1, k2, k1+1});
            }
            if(i != stacks-1)
            {
                triangles.push_back({k1+1, k2, k2+1});
            }
        }
    }

    std::shared_ptr<Transform> tf = Transform::Create("temp");
    glm::vec3 translation = glm::vec3(0);
    struct InstanceAttribute
    {
        glm::mat4 m2w;
        glm::vec3 albedo;
        float metallic;
        float roughness;
        float ao;
    } instances[nInstance];

    int k = 0;
    for(int i=0; i<m_row; i++)
    {
        translation.y = radius*3*i;
        for(int j=0; j<m_col; j++)
        {
            translation.x = radius*3*j;
            instances[k].m2w = tf->SetTranslation(translation)->Get();
//             instances[k].albedo = { rand()%256/255.0f, rand()%256/255.0f, rand()%256/255.0f };
            instances[k].albedo = {1, 0, 0};
            instances[k].metallic = i/float(m_row);
            instances[k].roughness = std::clamp(j/float(m_col), 0.05f, 1.0f);
            instances[k].ao = 1;
            k++;
        }
    }

    Buffer::Layout layoutVertex = { {Buffer::Element::DataType::Float3, "a_Position", false}, 
                                    {Buffer::Element::DataType::Float3, "a_Normal", false}, 
                                    {Buffer::Element::DataType::Float2, "a_TexCoord", false} };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UShort} };
    Buffer::Layout layoutInstance = { {Buffer::Element::DataType::Mat4, "a_MS2WS", false, 1}, 
                                      {Buffer::Element::DataType::Float3, "a_Albedo", false, 1}, 
                                      {Buffer::Element::DataType::Float, "a_Metallic", false, 1}, 
                                      {Buffer::Element::DataType::Float, "a_Roughness", false, 1}, 
                                      {Buffer::Element::DataType::Float, "a_Ao", false, 1} };
    
    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(vertices.size()*sizeof(VertexAttribute), &vertices[0])->SetLayout(layoutVertex);
    std::shared_ptr<Buffer> indexBuffer = Buffer::CreateIndex(triangles.size()*sizeof(glm::i16vec3), &triangles[0])->SetLayout(layoutIndex);
    std::shared_ptr<Buffer> instanceBuffer = Buffer::CreateVertex(nInstance*sizeof(InstanceAttribute), instances)->SetLayout(layoutInstance);
    std::shared_ptr<Elsa::Mesh> meshSphere = Renderer::Resources::Create<Elsa::Mesh>("Sphere_PBR0")->Set(indexBuffer, {vertexBuffer, instanceBuffer});
    std::shared_ptr<Texture> texNormal = Renderer::Resources::Create<Texture2D>("NormalMap")->Load("/home/garra/study/dnn/assets/texture/rustediron1-alt2-Unreal-Engine/rustediron2_normal.png");
    std::shared_ptr<Texture> texAlbedo = Renderer::Resources::Create<Texture2D>("AlbedoMap")->Load("/home/garra/study/dnn/assets/texture/rustediron1-alt2-Unreal-Engine/rustediron2_basecolor.png");
    std::shared_ptr<Texture> texRoughness = Renderer::Resources::Create<Texture2D>("RoughnessMap")->Load("/home/garra/study/dnn/assets/texture/rustediron1-alt2-Unreal-Engine/rustediron2_roughness.png");
    std::shared_ptr<Texture> texMetallic = Renderer::Resources::Create<Texture2D>("MetallicMap")->Load("/home/garra/study/dnn/assets/texture/rustediron1-alt2-Unreal-Engine/rustediron2_metallic.png");
    std::shared_ptr<Texture> texAo = Renderer::Resources::Create<Texture2D>("AoMap")->Load("/home/garra/study/dnn/assets/texture/rustediron1-alt2-Unreal-Engine/ao.png"); 


    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("Sphere_PBR0");
    mtr->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("Transform"));
    mtr->SetUniformBuffer("Light", Renderer::Resources::Get<UniformBuffer>("LightPBR0"));
    mtr->SetUniform("u_Camera.Position", Renderer::Resources::Get<Material::Uniform>("CameraPosition"));
    mtr->SetTexture("u_NormalMap", texNormal);
    mtr->SetTexture("u_AlbedoMap", texAlbedo);
    mtr->SetTexture("u_RoughnessMap", texRoughness);
    mtr->SetTexture("u_MetallicMap", texMetallic);
    mtr->SetTexture("u_AoMap", texAo);
    m_eleSpherePBR0 = Renderer::Resources::Create<Renderer::Element>("Sphere_PBR0")->Set(meshSphere, mtr);
//     m_shaderSpherePBR0 = Renderer::Resources::Create<Shader>("PBR0")->Define("NUM_OF_POINTLIGHTS 4")->LoadFromFile("/home/garra/study/dnn/assets/shader/PBR0.glsl");
    m_shaderSpherePBR0 = Renderer::Resources::Create<Shader>("PBR0")->Define("NUM_OF_POINTLIGHTS 4|NORMAL_MAP|ALBEDO_MAP|ROUGHNESS_MAP|METALLIC_MAP|AO_MAP")->LoadFromFile("/home/garra/study/dnn/assets/shader/PBR0.glsl");
}

void LearnOpenGLLayer::_PrepareSphere(float radius, int subdivision)
{
    // 12 30 20
    // 42 120 80
    // 132 330 320
    // 462 1290 1280
    // 1752 5130 5120
    const float PI = 3.14159265;
    const float H_ANGLE = PI/180*72;    // 72 degree
    const float V_ANGLE = atanf(0.5f);  // 26.565 degree

//     int nTri = 20*pow(4, subdivision);
//     int nVtx = 12;
//     for(int i=0; i<subdivision; i++)
//     {
//         nVtx += 20*pow(4, i)+10;
//         INFO("{}", nVtx);
//     }
    int nVtx = 12;
    int nEdg = 30;
    int nTri = 20;
    std::vector<glm::vec3> vertices(nVtx);
    std::vector<glm::i16vec2> edges(nEdg);
    std::vector<glm::i16vec3> triangles(nTri);

    float hAngle1 = -(PI+H_ANGLE)*0.5f;
    float hAngle2 = -PI*0.5f;
    vertices[0] = glm::vec3(0, 0, radius);
    vertices[11] = glm::vec3(0, 0, -radius);
    float z = radius*sinf(V_ANGLE);
    float xy = radius*cosf(V_ANGLE);
    for(int i=1; i<=5; i++)
    {
        vertices[i] = { xy*cosf(hAngle1), xy*sinf(hAngle1), z };
        vertices[i+5] = { xy*cosf(hAngle2), xy*sinf(hAngle2), -z };
        hAngle1 += H_ANGLE;
        hAngle2 += H_ANGLE;
    }

    triangles = { {0,  1, 2}, {0,  2, 3}, {0,  3, 4}, {0,  4,  5}, { 0,  5, 1}, 
                  {1,  6, 2}, {2,  7, 3}, {3,  8, 4}, {4,  9,  5}, { 5, 10, 1}, 
                  {1, 10, 6}, {2,  5, 6}, {3,  6, 7}, {4,  7,  8}, { 5,  8, 9}, 
                  {6, 11, 7}, {7, 11, 8}, {8, 11, 9}, {9, 11, 10}, {10, 11, 6} };
    edges = { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 1}, 
              {1, 10}, {1, 6}, {2, 6}, {2, 7}, {3, 7}, {3, 8}, {4, 8}, {4, 9}, {5, 9}, {5, 10}, 
              {11, 6}, {11, 7}, {11, 8}, {11, 9}, {11, 10}, {6, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 6} };

    

    for(auto v : vertices)
    {
        INFO("{}", glm::to_string(v));
    }
//     INFO("{}", sizeof(glm::vec3));
//     INFO("{}", sizeof(glm::i16vec3));
//     STOP


    for(int i=0; i<subdivision; i++)
    {
        _Subdivision(vertices, triangles);
    }

    Buffer::Layout layoutVertex = { {Buffer::Element::DataType::Float3, "a_PositionMS", false} };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UShort} };
    
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(nVtx*sizeof(glm::vec3), &vertices[0])->SetLayout(layoutVertex);
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(nTri*sizeof(glm::i16vec3), &triangles[0])->SetLayout(layoutIndex);
    std::shared_ptr<Elsa::Mesh> meshSphere = Renderer::Resources::Create<Elsa::Mesh>("Sphere")->Set(ib, {vb});

    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("Sphere");
    mtr->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("Transform"));
    m_eleSphere = Renderer::Resources::Create<Renderer::Element>("Sphere")->Set(meshSphere, mtr);
    m_shaderSphere = Renderer::Resources::Create<Shader>("Sphere")->LoadFromFile("/home/garra/study/dnn/assets/shader/Sphere.glsl");
}

void LearnOpenGLLayer::_Subdivision(std::vector<glm::vec3>& vertices, std::vector<glm::i16vec3>& triangles)
{
    int nVtx = vertices.size();
    int nTri = triangles.size();
    int nTriNext = nTri*4;
    int nVtxNext = nVtx+nTri+10;

    std::vector<glm::vec3> verticesNext(nVtxNext);
    std::vector<glm::i16vec3> trianglesNext(nTriNext);

    vertices.swap(verticesNext);
    triangles.swap(trianglesNext);




}

void LearnOpenGLLayer::_PrepareSkybox()
{
    float vertices[] = 
    {
        -1, -1, 
        +1, -1, 
        +1, +1, 
        -1, +1,
    };
    unsigned char indices[] = { 0, 1, 2, 0, 2, 3 };
    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float2, "a_Position", false} };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Elsa::Mesh> mesh= Renderer::Resources::Create<Elsa::Mesh>("BackgroundPlane")->Set(ib, {vb});
    std::shared_ptr<Texture> tex = Renderer::Resources::Create<Texture2D>("Skybox");
    tex->Load("/home/garra/study/dnn/assets/texture/skybox/autumn-crossing_3.jpg");
    using MU = Material::Uniform;
    std::shared_ptr<MU> maNearCorners = Renderer::Resources::Create<MU>("NearCorners")->Set(MU::Type::Float3, 4);
    std::shared_ptr<MU> maFarCorners = Renderer::Resources::Create<MU>("FarCorners")->Set(MU::Type::Float3, 4);
    std::shared_ptr<MU> maCornersDirection = Renderer::Resources::Create<MU>("CornersDirection")->Set(MU::Type::Float3, 4);
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("Skybox")->SetTexture("u_Skybox", tex);
    mtr->SetUniform("u_NearCorners", maNearCorners)->SetUniform("u_FarCorners", maFarCorners);
    Renderer::Resources::Create<Renderer::Element>("Skybox")->Set(mesh, mtr);
    Renderer::Resources::Create<Shader>("Skybox")->LoadFromFile("/home/garra/study/dnn/assets/shader/Skybox.glsl");
}

void LearnOpenGLLayer::_PrepareOffscreenPlane()
{
    using MU = Material::Uniform;
    std::shared_ptr<MU> maLeftBottomTexCoord = Renderer::Resources::Create<MU>("LeftBottomTexCoord")->SetType(MU::Type::Float2);
    std::shared_ptr<MU> maRightTopTexCoord = Renderer::Resources::Create<MU>("RightTopTexCoord")->SetType(MU::Type::Float2);
    std::shared_ptr<MU> maPostProcess = Renderer::Resources::Create<MU>("PostProcess")->Set(MU::Type::Int1, 1, &m_pp);
    std::shared_ptr<MU> maGamma = Renderer::Resources::Create<MU>("Gamma")->SetType(MU::Type::Float1);
    std::shared_ptr<MU> maExposure = Renderer::Resources::Create<MU>("Exposure")->SetType(MU::Type::Float1);
    int horizontal = 1;
    std::shared_ptr<MU> maHorizontalBlur = Renderer::Resources::Create<MU>("HorizontalBlur")->Set(MU::Type::Int1, 1, &horizontal);
    horizontal = 0;
    std::shared_ptr<MU> maVerticalBlur = Renderer::Resources::Create<MU>("VerticalBlur")->Set(MU::Type::Int1, 1, &horizontal);
    m_material_HDR.gamma = reinterpret_cast<float*>(maGamma->GetData());
    m_material_HDR.exposure = reinterpret_cast<float*>(maExposure->GetData());
    *m_material_HDR.gamma = 2.2f;
    *m_material_HDR.exposure = 1.0f;
    std::shared_ptr<Material>  mtrBase = Renderer::Resources::Create<Material>("Base");
    mtrBase->SetUniform("u_LeftBottomTexCoord", maLeftBottomTexCoord);
    mtrBase->SetUniform("u_RightTopTexCoord", maRightTopTexCoord);
    mtrBase->SetUniform("u_PostProcess", maPostProcess);
    mtrBase->SetUniform("u_Gamma", maGamma);
    mtrBase->SetUniform("u_Exposure", maExposure);
    mtrBase->SetTexture("u_Offscreen", m_texOffscreenBasic);
    std::shared_ptr<Material>  mtrBright = Renderer::Resources::Create<Material>("Bright");
    mtrBright->SetTexture("u_Offscreen", m_texOffscreenBright);
    std::shared_ptr<Material>  mtrBlurH = Renderer::Resources::Create<Material>("BlurH");
    mtrBlurH->SetUniform("u_LeftBottomTexCoord", maLeftBottomTexCoord);
    mtrBlurH->SetUniform("u_RightTopTexCoord", maRightTopTexCoord);
    mtrBlurH->SetTexture("u_Offscreen", m_texOffscreenBlurPong);
    mtrBlurH->SetUniform("u_Horizontal", maHorizontalBlur);
    std::shared_ptr<Material>  mtrBlurV = Renderer::Resources::Create<Material>("BlurV");
    mtrBlurV->SetUniform("u_LeftBottomTexCoord", maLeftBottomTexCoord); 
    mtrBlurV->SetUniform("u_RightTopTexCoord", maRightTopTexCoord);
    mtrBlurV->SetTexture("u_Offscreen", m_texOffscreenBlurPing);
    mtrBlurV->SetUniform("u_Horizontal", maVerticalBlur);

    std::shared_ptr<Material> mtrBloom = Renderer::Resources::Create<Material>("Bloom");
    mtrBloom->SetUniform("u_LeftBottomTexCoord", maLeftBottomTexCoord); 
    mtrBloom->SetUniform("u_RightTopTexCoord", maRightTopTexCoord);
    mtrBloom->SetTexture("u_Basic", m_texOffscreenBasic);
    mtrBloom->SetTexture("u_BrightBlur", m_texOffscreenBlurPong);
    mtrBloom->SetTexture("u_Offscreen", m_texOffscreenBloom);


    m_leftBottomTexCoord = reinterpret_cast<glm::vec2*>(maLeftBottomTexCoord->GetData());
    m_rightTopTexCoord = reinterpret_cast<glm::vec2*>(maRightTopTexCoord->GetData());
    std::array<float, 4> r = m_vpOffscreen->GetRange();
    float w = m_offscreenBufferSize.x;
    float h = m_offscreenBufferSize.y;
    *m_leftBottomTexCoord = glm::vec2(r[0]/w, r[1]/h);
    *m_rightTopTexCoord = glm::vec2((r[0]+r[2])/w, (r[1]+r[3])/h);


    float vertices[] = 
    {
        -1, -1, 
        +1, -1, 
        +1, +1,
        -1, +1, 
    };
    unsigned char indices[] = { 0, 1, 2, 0, 2, 3 };
    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float2, "a_Position", false} };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Elsa::Mesh> mesh= Renderer::Resources::Create<Elsa::Mesh>("OffscreenPlane")->Set(ib, {vb});

    m_eleBase = Renderer::Resources::Create<Renderer::Element>("Base")->Set(mesh, mtrBase);
    m_eleBright = Renderer::Resources::Create<Renderer::Element>("Bright")->Set(mesh, mtrBright);
    m_eleBlurH = Renderer::Resources::Create<Renderer::Element>("BlurH")->Set(mesh, mtrBlurH);
    m_eleBlurV = Renderer::Resources::Create<Renderer::Element>("BlurV")->Set(mesh, mtrBlurV);
    m_eleBloom = Renderer::Resources::Create<Renderer::Element>("Bloom")->Set(mesh, mtrBloom);


    m_shaderHDR = Renderer::Resources::Create<Shader>(_StringOfShaderID_HDR())->Define(m_shaderID_HDR)->LoadFromFile("/home/garra/study/dnn/assets/shader/OffscreenTexture.glsl");
    m_shaderBlur = Renderer::Resources::Create<Shader>("Blur")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blur.glsl");
    m_shaderBloom = Renderer::Resources::Create<Shader>("Bloom")->Define("BLOOM|TONE_MAP")->LoadFromFile("/home/garra/study/dnn/assets/shader/Bloom.glsl");
}

void LearnOpenGLLayer::_PrepareGroundPlane()
{
    using MU = Material::Uniform;
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("GroundPlane");
    mtr->SetUniform("u_NearCorners", Renderer::Resources::Get<MU>("NearCorners"));
    mtr->SetUniform("u_FarCorners", Renderer::Resources::Get<MU>("FarCorners"));
    mtr->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("Transform"));
    Renderer::Resources::Create<Renderer::Element>("GroundPlane")->SetMesh("BackgroundPlane")->SetMaterial(mtr);
    Renderer::Resources::Create<Shader>("GroundPlane")->LoadFromFile("/home/garra/study/dnn/assets/shader/GroundPlane.glsl");
}

void LearnOpenGLLayer::_UpdateMaterialUniforms()
{
    using MU = Material::Uniform;
    const std::shared_ptr<Camera>& cam = m_vpBase->GetCamera();
    Renderer::Resources::Get<MU>("CameraPosition")->UpdateData(&cam->GetPosition());
    Renderer::Resources::Get<MU>("NearCorners")->UpdateData(&cam->GetNearCornersInWorldSpace()[0]);
    Renderer::Resources::Get<MU>("FarCorners")->UpdateData(&cam->GetFarCornersInWorldSpace()[0]);
    Renderer::Resources::Get<UniformBuffer>("Transform")->Upload("WS2CS", glm::value_ptr(m_vpBase->GetCamera()->World2Clip()));
    m_fLight.pos = glm::vec4(cam->GetPosition(), 1);
    m_fLight.dir = glm::vec4(cam->GetDirection(), 0);
    Renderer::Resources::Get<UniformBuffer>("Light")->Upload("FlashLight", &m_fLight);
}

void LearnOpenGLLayer::_PrepareUniformBuffers()
{
    std::shared_ptr<UniformBuffer> ubTransform = Renderer::Resources::Create<UniformBuffer>("Transform")->SetSize(64);
    ubTransform->Push("WS2CS", glm::ivec2(0, 64));
    ubTransform->Upload("WS2CS", glm::value_ptr(m_vpBase->GetCamera()->World2Clip()));

    std::shared_ptr<UniformBuffer> ubLight = Renderer::Resources::Create<UniformBuffer>("Light")->SetSize(240);
    ubLight->Push("DirectionalLight", glm::ivec2(0, 32));
    ubLight->Push("PointLight", glm::ivec2(32, 48));
    ubLight->Push("SpotLight", glm::ivec2(80, 64));
    ubLight->Push("FlashLight", glm::ivec2(144, 64));
    ubLight->Upload("DirectionalLight", &m_dLight);
    ubLight->Upload("PointLight", &m_pLight);
    ubLight->Upload("SpotLight", &m_sLight);
    ubLight->Upload("FlashLight", &m_fLight);

    std::shared_ptr<UniformBuffer> ubLightPBR0 = Renderer::Resources::Create<UniformBuffer>("LightPBR0")->SetSize(64*4);
    ubLightPBR0->Push("AllLights", glm::ivec2(0, 64*4));
    struct PointLight
    {
        glm::vec3 pos;
        float padding0;
        glm::vec3 clr;
        float padding1;
    }
    lights[4];

    lights[0].pos = glm::vec3(3, 3, 6);
    lights[0].clr = glm::vec3(1);
    lights[1].pos = glm::vec3(9, 3, 6);
    lights[1].clr = glm::vec3(1);
    lights[2].pos = glm::vec3(3, 9, 6);
    lights[2].clr = glm::vec3(1);
    lights[3].pos = glm::vec3(9, 9, 6);
    lights[3].clr = glm::vec3(1);
    ubLightPBR0->Upload("AllLights", lights);
}

