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

std::shared_ptr<LearnOpenGLLayer> LearnOpenGLLayer::Create()
{
    return std::make_shared<LearnOpenGLLayer>();
}

LearnOpenGLLayer::LearnOpenGLLayer()
    : Layer( "LearnOpenGLLayer" )
{
    m_viewport->GetCamera()->SetFrameBuffer(m_fbSS);
    m_viewport->GetCamera()->SetTarget(glm::vec3(0, 8, 0));
    _PrepareUniformBuffers();
    _PrepareSkybox();
    _PrepareOffscreenPlane();
    _PrepareUnitCubic();
    _PrepareGroundPlane();
    _PrepareModel();
}

void LearnOpenGLLayer::OnEvent(Event& e)
{
    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(BIND_EVENT_CALLBACK(LearnOpenGLLayer, _On##event))
    DISPATCH(WindowResizeEvent);
#undef DISPATCH
    m_viewport->OnEvent(e);
}

bool LearnOpenGLLayer::_OnWindowResizeEvent(WindowResizeEvent& e)
{
    m_rightTopTexCoord->x = e.GetWidth()/(float)m_fbSS->GetWidth();
    m_rightTopTexCoord->y = e.GetHeight()/(float)m_fbSS->GetHeight();
    return false;
}

void LearnOpenGLLayer::OnUpdate(float deltaTime)
{
    _UpdateMaterialUniforms();

    Renderer::BeginScene(m_viewport, m_fbMS);
//     Renderer::BeginScene(m_viewport);
//     m_crysisNanoSuit->Draw(m_shaderPos);
    m_crysisNanoSuit->Draw(m_shaderBlinnPhong);
//     m_trailer->Draw(m_shaderBlinnPhong);
//         m_silkingMachine->Draw(m_shaderBlinnPhong);
//         m_horse->Draw(m_shaderBlinnPhong);
//     m_bulb->Draw(m_shaderColor);
//     m_handLight->Draw(m_shaderColor);
    if(m_showGround)
        Renderer::Submit("GroundPlane", "GroundPlane");

    if(m_showSky)
        Renderer::Submit("Skybox", "Skybox");

    Renderer::Submit("UnitCubic", "Blinn-Phong-Instance", m_numOfInstance);
    Renderer::EndScene();

    Renderer::BlitFrameBuffer(m_fbMS, m_fbSS);

    Renderer::BeginScene(m_viewport);
    Renderer::Submit("Offscreen", "Offscreen");
    Renderer::EndScene();
 
    m_viewport->OnUpdate(deltaTime);
}

void LearnOpenGLLayer::OnImGuiRender()
{
    using MU = Material::Uniform;
    m_viewport->OnImGuiRender();

    ImGui::Begin("LearnOpenGLLayer");

    if(ImGui::RadioButton("ShowSky", m_showSky))
    {
        m_showSky = !m_showSky;
    }

    ImGui::SameLine();
    if(ImGui::RadioButton("ShowGround", m_showGround))
    {
        m_showGround = !m_showGround;
    }

    ImGui::Separator();
    if(ImGui::InputInt("Samples", (int*)&m_samples))
    {
        unsigned int w = m_fbMS->GetWidth();
        unsigned int h = m_fbMS->GetHeight();
        m_fbMS->Reset(w, h, m_samples);
    }
    if(ImGui::CollapsingHeader("PostProcess"))
    {
#define RadioButton(x) \
        if(ImGui::RadioButton(#x, m_pp == PostProcess::x)) \
        {                                                  \
            m_pp = PostProcess::x;                         \
            Renderer::Resources::Get<MU>("PostProcess")->UpdateData(&m_pp); \
        }                                                                   \

        RadioButton(None);
        RadioButton(Gray);
        RadioButton(Smooth);
        RadioButton(Edge);
#undef RadioButton
    }

    ImGui::Separator();
    ImGui::PushItemWidth(120);
    if(ImGui::CollapsingHeader("Environment"))
    {
        ImGui::ColorPicker3("AmbientColor", reinterpret_cast<float*>(m_ambientColor));
    }

    if(ImGui::CollapsingHeader("Material"))
    {
        ImGui::ColorPicker3("AmbientReflectance", reinterpret_cast<float*>(m_material.ambientReflectance));
        ImGui::ColorPicker3("DiffuseReflectance", reinterpret_cast<float*>(m_material.diffuseReflectance));
        ImGui::ColorPicker3("SpecularReflectance", reinterpret_cast<float*>(m_material.specularReflectance));
        ImGui::ColorPicker3("EmissiveColor", reinterpret_cast<float*>(m_material.emissiveColor));
        ImGui::SliderFloat("Shininess", m_material.shininess, 0, 512);
    }

    if(ImGui::CollapsingHeader("Light"))
    {
        ImGui::SetNextWindowPos({100, 0});
        if(ImGui::CollapsingHeader("DirectionalLight"))
        {
            ImGui::ColorPicker3("Color", reinterpret_cast<float*>(m_directionalLight.color));
            ImGui::SliderFloat3("Direction", reinterpret_cast<float*>(m_directionalLight.direction), -1, 1);
        }
        if(ImGui::CollapsingHeader("PointLight"))
        {
            ImGui::ColorPicker3("Color", reinterpret_cast<float*>(m_pointLight.color));
            ImGui::SliderFloat3("Position", reinterpret_cast<float*>(m_pointLight.position), -10, 10);
            ImGui::SliderFloat3("AttenuationCoefficents", reinterpret_cast<float*>(m_pointLight.attenuationCoefficients), -10, 10);
        }
        if(ImGui::CollapsingHeader("SpotLight"))
        {
            ImGui::ColorPicker3("Color", reinterpret_cast<float*>(m_spotLight.color));
            ImGui::SliderFloat3("Position", reinterpret_cast<float*>(m_spotLight.position), -10, 10);
            ImGui::SliderFloat3("AttenuationCoefficents", reinterpret_cast<float*>(m_spotLight.attenuationCoefficients), -10, 10);
            ImGui::SliderFloat3("Direction", reinterpret_cast<float*>(m_spotLight.direction), -1, 1);
            if(ImGui::SliderFloat("InnerCone", &m_spotLight.innerCone, 0, 90))
            {
                float innerCone = std::cos(glm::radians(m_spotLight.innerCone));
                Renderer::Resources::Get<MU>("SLightInnerCone")->UpdateData(&innerCone);
            }
            if(ImGui::SliderFloat("OuterCone", &m_spotLight.outerCone, 0, 120))
            {
                float outerCone = std::cos(glm::radians(m_spotLight.outerCone));
                Renderer::Resources::Get<MU>("SLightOuterCone")->UpdateData(&outerCone);
            }
        }
        if(ImGui::CollapsingHeader("FlashLight"))
        {
            ImGui::ColorPicker3("Color", reinterpret_cast<float*>(m_flashLight.color));
            ImGui::SliderFloat3("AttenuationCoefficents", reinterpret_cast<float*>(m_flashLight.attenuationCoefficients), -10, 10);
            if(ImGui::SliderFloat("InnerCone", &m_flashLight.innerCone, 0, 90))
            {
                float innerCone = std::cos(glm::radians(m_flashLight.innerCone));
                Renderer::Resources::Get<MU>("FLightInnerCone")->UpdateData(&innerCone);
            }
            if(ImGui::SliderFloat("OuterCone", &m_flashLight.outerCone, 0, 120))
            {
                float outerCone = std::cos(glm::radians(m_flashLight.outerCone));
                Renderer::Resources::Get<MU>("FLightOuterCone")->UpdateData(&outerCone);
            }
        }
    }

    ImGui::End();
}

void LearnOpenGLLayer::_PrepareModel()
{
    m_crysisNanoSuit = Renderer::Resources::Create<Model>("CysisNanoSuit")->LoadFromFile("/home/garra/study/dnn/assets/mesh/CysisNanoSuit/scene.fbx");
//     m_silkingMachine = Renderer::Resources::Create<Model>("SilkingMachine")->LoadFromFile("/home/garra/study/dnn/assets/mesh/SilkingMachine/SilkingMachine.fbx");
//     m_horse = Renderer::Resources::Create<Model>("Horse")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Horse/Horse.fbx");
//     m_trailer = Renderer::Resources::Create<Model>("Trailer")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Trailer/Alena_Shek.obj");
//     m_bulb = Renderer::Resources::Create<Model>("Bulb")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Bulb/Bulbs.3ds");
//     m_handLight = Renderer::Resources::Create<Model>("HandLight")->LoadFromFile("/home/garra/study/dnn/assets/mesh/HandLight/hand_light.blend");
//     m_shaderPos = Renderer::Resources::Create<Shader>("Pos")->LoadFromFile("/home/garra/study/dnn/assets/shader/Model.glsl");
//     Renderer::Resources::Create<Shader>("Default")->LoadFromFile("/home/garra/study/dnn/assets/shader/Default.glsl");
}

void LearnOpenGLLayer::_PrepareUnitCubic()
{
    float vertices[] = 
    {
        // front
        -1, -1, +1, 0, 0, +1, 0, 0,   
        +1, -1, +1, 0, 0, +1, 1, 0, 
        +1, +1, +1, 0, 0, +1, 1, 1, 
        -1, +1, +1, 0, 0, +1, 0, 1, 
        // back
        +1, -1, -1, 0, 0, -1, 0, 0, 
        -1, -1, -1, 0, 0, -1, 1, 0, 
        -1, +1, -1, 0, 0, -1, 1, 1, 
        +1, +1, -1, 0, 0, -1, 0, 1, 
        // left
        -1, -1, -1, -1, 0, 0, 0, 0, 
        -1, -1, +1, -1, 0, 0, 1, 0, 
        -1, +1, +1, -1, 0, 0, 1, 1, 
        -1, +1, -1, -1, 0, 0, 0, 1, 
        // right
        +1, -1, +1, +1, 0, 0, 0, 0, 
        +1, -1, -1, +1, 0, 0, 1, 0, 
        +1, +1, -1, +1, 0, 0, 1, 1, 
        +1, +1, +1, +1, 0, 0, 0, 1, 
        // up
        -1, +1, +1, 0, +1, 0, 0, 0, 
        +1, +1, +1, 0, +1, 0, 1, 0, 
        +1, +1, -1, 0, +1, 0, 1, 1, 
        -1, +1, -1, 0, +1, 0, 0, 1, 
        // down
        -1, -1, -1, 0, -1, 0, 0, 0, 
        +1, -1, -1, 0, -1, 0, 1, 0, 
        +1, -1, +1, 0, -1, 0, 1, 1, 
        -1, -1, +1, 0, -1, 0, 0, 1, 
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

    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float3, "a_Position", false}, 
                                    {Buffer::Element::DataType::Float3, "a_Normal", false},
                                    {Buffer::Element::DataType::Float2, "a_TexCoord", false}  };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    Buffer::Layout layoutInstance = { {Buffer::Element::DataType::Mat4, "a_Model2World", false, 1} };

    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> indexBuffer = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Buffer> instanceBuffer = Buffer::CreateVertex(m_numOfInstance*sizeof(glm::mat4), matM2W)->SetLayout(layoutInstance);
    std::shared_ptr<Elsa::Mesh> mesh= Renderer::Resources::Create<Elsa::Mesh>("UnitCubic")->Set(indexBuffer, {vertexBuffer, instanceBuffer});
    using MU = Material::Uniform;


    std::shared_ptr<MU> maMaterialAmbientReflectance = Renderer::Resources::Create<MU>("MaterialAmbientReflectance")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1f)));
    std::shared_ptr<MU> maMaterialDiffuseReflectance = Renderer::Resources::Create<MU>("MaterialDiffuseReflectance")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.5f)));
    std::shared_ptr<MU> maMaterialSpecularReflectance = Renderer::Resources::Create<MU>("MaterialSpecularReflectance")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0f)));
    std::shared_ptr<MU> maMaterialEmissiveColor = Renderer::Resources::Create<MU>("MaterialEmissiveColor")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1f)));
    std::shared_ptr<MU> maMaterialShininess = Renderer::Resources::Create<MU>("MaterialShininess")->SetType(MU::Type::Float1);

    std::shared_ptr<MU> maCameraPosition = Renderer::Resources::Create<MU>("CameraPosition")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(2.0f)));


    m_material.ambientReflectance = reinterpret_cast<glm::vec3*>(maMaterialAmbientReflectance->GetData());
    m_material.diffuseReflectance = reinterpret_cast<glm::vec3*>(maMaterialDiffuseReflectance->GetData());
    m_material.specularReflectance = reinterpret_cast<glm::vec3*>(maMaterialSpecularReflectance->GetData());
    m_material.emissiveColor = reinterpret_cast<glm::vec3*>(maMaterialEmissiveColor->GetData());
    m_material.shininess = reinterpret_cast<float*>(maMaterialShininess->GetData());
    m_material.diffuseMap = Renderer::Resources::Create<Texture2D>("DiffuseMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/container2.png");
    m_material.specularMap = Renderer::Resources::Create<Texture2D>("SpecularMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/lighting_maps_specular_color.png");
    m_material.emissiveMap = Renderer::Resources::Create<Texture2D>("EmissiveMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/matrix.jpg");
    *m_material.shininess = 32.0f;


    // AmbientColor
    std::shared_ptr<MU> maAmbientColor = Renderer::Resources::Create<MU>("AmbientColor")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1f)));
    m_ambientColor = reinterpret_cast<glm::vec3*>(maAmbientColor->GetData());

    // DirectionalLight
    std::shared_ptr<MU> maDLightColor = Renderer::Resources::Create<MU>("DLightColor")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0f)));
    std::shared_ptr<MU> maDLightDirection = Renderer::Resources::Create<MU>("DLightDirection")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(-1, -2, -3)));
    m_directionalLight.color = reinterpret_cast<glm::vec3*>(maDLightColor->GetData());
    m_directionalLight.direction = reinterpret_cast<glm::vec3*>(maDLightDirection->GetData());

    // PointLight
    std::shared_ptr<MU> maPLightColor = Renderer::Resources::Create<MU>("PLightColor")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1, 0.8, 0.2)));
    std::shared_ptr<MU> maPLightPosition = Renderer::Resources::Create<MU>("PLightPosition")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0, 2, 0)));
    std::shared_ptr<MU> maPLightCoefs = Renderer::Resources::Create<MU>("PLightCoefs")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0, 1.0, 2.0)));
    m_pointLight.color = reinterpret_cast<glm::vec3*>(maPLightColor->GetData());
    m_pointLight.position = reinterpret_cast<glm::vec3*>(maPLightPosition->GetData());
    m_pointLight.attenuationCoefficients = reinterpret_cast<glm::vec3*>(maPLightCoefs->GetData());

    // SpotLight
    std::shared_ptr<MU> maSLightColor = Renderer::Resources::Create<MU>("SLightColor")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1, 0.1, 0.9)));
    std::shared_ptr<MU> maSLightPosition = Renderer::Resources::Create<MU>("SLightPosition")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0, 0, 2)));
    std::shared_ptr<MU> maSLightCoefs = Renderer::Resources::Create<MU>("SLightCoefs")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0, 1.0, 2.0)));
    std::shared_ptr<MU> maSLightDirection = Renderer::Resources::Create<MU>("SLightDirection")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0, 0, -1)));
    std::shared_ptr<MU> maSLightInnerCone = Renderer::Resources::Create<MU>("SLightInnerCone")->Set(MU::Type::Float1, 1, &m_spotLight.innerCone);
    std::shared_ptr<MU> maSLightOuterCone = Renderer::Resources::Create<MU>("SLightOuterCone")->Set(MU::Type::Float1, 1, &m_spotLight.outerCone);
    m_spotLight.color = reinterpret_cast<glm::vec3*>(maSLightColor->GetData());
    m_spotLight.position = reinterpret_cast<glm::vec3*>(maSLightPosition->GetData());
    m_spotLight.attenuationCoefficients = reinterpret_cast<glm::vec3*>(maSLightCoefs->GetData());
    m_spotLight.direction = reinterpret_cast<glm::vec3*>(maSLightDirection->GetData());

    // FlashLight
    std::shared_ptr<MU> maFLightColor = Renderer::Resources::Create<MU>("FLightColor")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.8, 0.1, 0.2)));
    std::shared_ptr<MU> maFLightPosition = Renderer::Resources::Create<MU>("FLightPosition")->Set(MU::Type::Float3, 1, glm::value_ptr(m_viewport->GetCamera()->GetPosition()));
    std::shared_ptr<MU> maFLightCoefs = Renderer::Resources::Create<MU>("FLightCoefs")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0, 1.0, 2.0)));
    std::shared_ptr<MU> maFLightDirection = Renderer::Resources::Create<MU>("FLightDirection")->Set(MU::Type::Float3, 1, glm::value_ptr(m_viewport->GetCamera()->GetDirection()));
    std::shared_ptr<MU> maFLightInnerCone = Renderer::Resources::Create<MU>("FLightInnerCone")->Set(MU::Type::Float1, 1, &m_flashLight.innerCone);
    std::shared_ptr<MU> maFLightOuterCone = Renderer::Resources::Create<MU>("FLightOuterCone")->Set(MU::Type::Float1, 1, &m_flashLight.outerCone);
    m_flashLight.color = reinterpret_cast<glm::vec3*>(maFLightColor->GetData());
    m_flashLight.attenuationCoefficients = reinterpret_cast<glm::vec3*>(maFLightCoefs->GetData());

    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("UnitCubic");
    mtr->SetUniform("u_Material.ambientReflectance", maMaterialAmbientReflectance);
    mtr->SetUniform("u_Material.diffuseReflectance", maMaterialDiffuseReflectance);
    mtr->SetUniform("u_Material.specularReflectance", maMaterialSpecularReflectance);
    mtr->SetUniform("u_Material.emissiveColor", maMaterialEmissiveColor);
    mtr->SetUniform("u_Material.shininess", maMaterialShininess);
    mtr->SetTexture("u_Material.diffuseMap", m_material.diffuseMap);
    mtr->SetTexture("u_Material.specularMap", m_material.specularMap);
    mtr->SetTexture("u_Material.emissiveMap", m_material.emissiveMap);

    // AmbientColor
    mtr->SetUniform("u_AmbientColor", maAmbientColor);

    // DirectionalLight
    mtr->SetUniform("u_DirectionalLight[0].color", maDLightColor);
    mtr->SetUniform("u_DirectionalLight[0].direction", maDLightDirection);

    // PointLight
    mtr->SetUniform("u_PointLight[0].position", maPLightPosition);
    mtr->SetUniform("u_PointLight[0].color", maPLightColor);
    mtr->SetUniform("u_PointLight[0].attenuationCoefficients", maPLightCoefs);

    // SpotLight
    mtr->SetUniform("u_SpotLight[0].position", maSLightPosition);
    mtr->SetUniform("u_SpotLight[0].direction", maSLightDirection);
    mtr->SetUniform("u_SpotLight[0].attenuationCoefficients", maSLightCoefs);
    mtr->SetUniform("u_SpotLight[0].color", maSLightColor);
    mtr->SetUniform("u_SpotLight[0].innerCone", maSLightInnerCone);
    mtr->SetUniform("u_SpotLight[0].outerCone", maSLightOuterCone);
    
    // FlashLight
    mtr->SetUniform("u_FlashLight.position", maFLightPosition);
    mtr->SetUniform("u_FlashLight.direction", maFLightDirection);
    mtr->SetUniform("u_FlashLight.attenuationCoefficients", maFLightCoefs);
    mtr->SetUniform("u_FlashLight.color", maFLightColor);
    mtr->SetUniform("u_FlashLight.innerCone", maFLightInnerCone);
    mtr->SetUniform("u_FlashLight.outerCone", maFLightOuterCone);

    mtr->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("Transform"));
    mtr->SetUniformBuffer("Light", Renderer::Resources::Get<UniformBuffer>("Light"));

    Renderer::Resources::Create<Shader>("Blinn-Phong-Instance")->Define("_INSTANCE_")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");
    m_shaderBlinnPhong = Renderer::Resources::Create<Shader>("Blinn-Phong")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");

    Renderer::Resources::Create<Renderer::Element>("UnitCubic")->Set(mesh, mtr);


    unsigned int id = m_shaderBlinnPhong->ID();
    GLuint index = glGetUniformBlockIndex(id, "Light");
    INFO("Light: index {}", index);
    if(GL_INVALID_INDEX != index)
    {
        GLint size = 0;
        glGetActiveUniformBlockiv(id, index, GL_UNIFORM_BLOCK_DATA_SIZE, &size);
        INFO("BlockDataSize: {}", size);
    }

//     STOP
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
    tex->LoadFromFile("/home/garra/study/dnn/assets/texture/skybox/autumn-crossing_3.jpg");
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
    std::shared_ptr<MU> maRightTopTexCoord = Renderer::Resources::Create<MU>("RightTopTexCoord")->SetType(MU::Type::Float2);
    std::shared_ptr<MU> maPostProcess = Renderer::Resources::Create<MU>("PostProcess")->Set(MU::Type::Int1, 1, &m_pp);
    std::shared_ptr<Material>  mtr = Renderer::Resources::Create<Material>("Offscreen");
    mtr->SetUniform("u_RightTopTexCoord", maRightTopTexCoord);
    mtr->SetUniform("u_PostProcess", maPostProcess);
    mtr->SetTexture("u_Offscreen", m_fbSS->GetColorBuffer());
    m_rightTopTexCoord = reinterpret_cast<glm::vec2*>(maRightTopTexCoord->GetData());
    *m_rightTopTexCoord = glm::vec2(1, 1);
    std::array<float, 4> r = m_viewport->GetRange();
    *m_rightTopTexCoord = glm::vec2(r[2]/m_fbSS->GetWidth(), r[3]/m_fbSS->GetHeight());

    float vertices[] = 
    {
        -1, -1, 
        +1, -1, 
        -1, +1, 
        +1, +1,
    };
    unsigned char indices[] = { 0, 1, 3, 0, 3, 2 };
    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float2, "a_Position", false} };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Elsa::Mesh> mesh= Renderer::Resources::Create<Elsa::Mesh>("OffscreenPlane")->Set(ib, {vb});

    Renderer::Resources::Create<Renderer::Element>("Offscreen")->Set(mesh, mtr);
    Renderer::Resources::Create<Shader>("Offscreen")->LoadFromFile("/home/garra/study/dnn/assets/shader/OffscreenTexture.glsl");
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
    const std::shared_ptr<Camera>& cam = m_viewport->GetCamera();
    Renderer::Resources::Get<MU>("CameraPosition")->UpdateData(&cam->GetPosition());
    Renderer::Resources::Get<MU>("NearCorners")->UpdateData(&cam->GetNearCornersInWorldSpace()[0]);
    Renderer::Resources::Get<MU>("FarCorners")->UpdateData(&cam->GetFarCornersInWorldSpace()[0]);
    Renderer::Resources::Get<MU>("FLightPosition")->UpdateData(&cam->GetPosition());
    glm::vec3 dir = cam->GetDirection();
    Renderer::Resources::Get<MU>("FLightDirection")->UpdateData(&dir);
    Renderer::Resources::Get<UniformBuffer>("Transform")->Upload("World2Clip", glm::value_ptr(m_viewport->GetCamera()->World2Clip()));
}

void LearnOpenGLLayer::_PrepareUniformBuffers()
{
    std::shared_ptr<UniformBuffer> ubTransform = Renderer::Resources::Create<UniformBuffer>("Transform")->SetSize(64);
    ubTransform->Push("World2Clip", glm::vec2(0, 64));
    ubTransform->Upload("World2Clip", glm::value_ptr(m_viewport->GetCamera()->World2Clip()));

    std::shared_ptr<UniformBuffer> ubLight = Renderer::Resources::Create<UniformBuffer>("Light")->SetSize(240);
    ubLight->Push("DirectionalLight", glm::vec2(0, 32));
    ubLight->Push("PointLight", glm::vec2(32, 48));
    ubLight->Push("SpotLight", glm::vec2(80, 80));
    ubLight->Push("FlashLight", glm::vec2(160, 80));

    struct Light
    {
        glm::vec4 clr;
    };

    struct DirectionalLight : public Light
    {
        glm::vec4 dir;
    };

    struct PointLight : public Light
    {
        glm::vec4 pos;
        glm::vec4 coe;
    };

    struct SpotLight : public Light
    {
        glm::vec4 dir;
        glm::vec4 pos;
        glm::vec4 coe;
        float innerCone;
        float outerCone;
        float pad[2];
    };

    DirectionalLight dLight;
    PointLight pLight;
    SpotLight sLight;
    SpotLight fLight;
    
    dLight.clr = glm::vec4(1, 0, 0, 1);
    dLight.dir = glm::vec4(0, 0, -1, 0);
    pLight.clr = glm::vec4(0, 1, 0, 1);
    pLight.pos = glm::vec4(0, 5, 10, 1);
    pLight.coe = glm::vec4(1.0, 0.09, 0.032, 0.0);
    sLight.clr = glm::vec4(0, 0, 1, 1);
    sLight.pos = glm::vec4(0, 5, -10, 1);
    sLight.dir = glm::vec4(0, 0, 1, 0);
    sLight.coe = glm::vec4(1.0, 0.22, 0.20, 0.0);
    sLight.innerCone = std::cos(glm::radians(30.0));
    sLight.outerCone = std::cos(glm::radians(45.0));
    fLight.clr = glm::vec4(1, 1, 1, 1);
    fLight.pos = glm::vec4(m_viewport->GetCamera()->GetPosition(), 1);
    fLight.dir = glm::vec4(m_viewport->GetCamera()->GetDirection(), 0);
    fLight.coe = glm::vec4(1.0, 0.14, 0.07, 0.0);
    fLight.innerCone = std::cos(glm::radians(20.0));
    fLight.outerCone = std::cos(glm::radians(30.0));

    ubLight->Upload("DirectionalLight", &dLight);
    ubLight->Upload("PointLight", &pLight);
    ubLight->Upload("SpotLight", &sLight);
    ubLight->Upload("FlashLight", &fLight);
}
