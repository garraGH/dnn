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
    m_viewport->GetCamera()->SetPosition(glm::vec3(0, 1, 5));
    m_viewport->GetCamera()->SetTarget(glm::vec3(0));

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
//     m_crysisNanoSuit->Draw(m_shaderBlinnPhong);
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
    Renderer::Submit("UnitCubic", m_bUseNormalMap? "BlinnWithDiffuseNormalMap" : "BlinnWithDiffuseMap");
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
    if(ImGui::RadioButton("UseNormalMap", m_bUseNormalMap))
    {
        m_bUseNormalMap = !m_bUseNormalMap;
    }

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
    ImGui::PushItemWidth(200);
    if(ImGui::CollapsingHeader("Environment"))
    {
        ImGui::ColorPicker3("AmbientColor", reinterpret_cast<float*>(m_ambientColor));
    }

    if(ImGui::CollapsingHeader("Material"))
    {
        ImGui::ColorPicker3("DiffuseReflectance", reinterpret_cast<float*>(m_material.diffuseReflectance));
        ImGui::ColorPicker3("EmissiveColor", reinterpret_cast<float*>(m_material.emissiveColor));
        ImGui::ColorPicker3("SpecularReflectance", reinterpret_cast<float*>(m_material.specularReflectance));
        ImGui::SliderFloat("Shininess", m_material.shininess, 0, 512);
    }

//     Renderer::Resources::Get<UniformBuffer>("Light")->Upload(name, data)
    if(ImGui::CollapsingHeader("Light"))
    {
        ImGui::SetNextWindowPos({200, 0});
        if(ImGui::CollapsingHeader("DirectionalLight"))
        {
            bool bChanged = ImGui::ColorPicker3("Color", glm::value_ptr(m_dLight.clr));
            bChanged |= ImGui::DragFloat3("Direction", glm::value_ptr(m_dLight.dir), 0.1f, -1, 1);
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
    std::shared_ptr<MU> maMaterialShininess = Renderer::Resources::Create<MU>("MaterialShininess")->SetType(MU::Type::Float1);
    std::shared_ptr<MU> maMaterialDepthScale = Renderer::Resources::Create<MU>("MaterialDepthScale")->SetType(MU::Type::Float1);

    std::shared_ptr<MU> maCameraPosition = Renderer::Resources::Create<MU>("CameraPosition")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(2.0f)));


    m_material.diffuseReflectance = reinterpret_cast<glm::vec3*>(maMaterialDiffuseReflectance->GetData());
    m_material.specularReflectance = reinterpret_cast<glm::vec3*>(maMaterialSpecularReflectance->GetData());
    m_material.emissiveColor = reinterpret_cast<glm::vec3*>(maMaterialEmissiveColor->GetData());
    m_material.shininess = reinterpret_cast<float*>(maMaterialShininess->GetData());
    *m_material.shininess = 32.0f;
    m_material.depthScale = reinterpret_cast<float*>(maMaterialDepthScale->GetData());
    *m_material.depthScale = 0.1f;
    m_material.diffuseMap = Renderer::Resources::Create<Texture2D>("DiffuseMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/wood.png");
    m_material.normalMap = Renderer::Resources::Create<Texture2D>("NormalMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/toy_box_normal.png");
    m_material.depthMap = Renderer::Resources::Create<Texture2D>("DepthMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/toy_box_disp.png");
//     m_material.diffuseMap = Renderer::Resources::Create<Texture2D>("DiffuseMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/bricks2.jpg");
//     m_material.normalMap = Renderer::Resources::Create<Texture2D>("NormalMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/bricks2_normal.jpg");
//     m_material.depthMap = Renderer::Resources::Create<Texture2D>("DepthMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/bricks2_disp.jpg");
//     m_material.diffuseMap = Renderer::Resources::Create<Texture2D>("DiffuseMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/container2.png");
//     m_material.specularMap = Renderer::Resources::Create<Texture2D>("SpecularMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/lighting_maps_specular_color.png");
//     m_material.emissiveMap = Renderer::Resources::Create<Texture2D>("EmissiveMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/matrix.jpg");


    // AmbientColor
    std::shared_ptr<MU> maAmbientColor = Renderer::Resources::Create<MU>("AmbientColor")->Set(MU::Type::Float3, 1, glm::value_ptr(glm::vec3(0.3f)));
    m_ambientColor = reinterpret_cast<glm::vec3*>(maAmbientColor->GetData());

    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("UnitCubic");
    mtr->SetUniform("u_Material.DiffuseReflectance", maMaterialDiffuseReflectance);
    mtr->SetUniform("u_Material.SpecularReflectance", maMaterialSpecularReflectance);
    mtr->SetUniform("u_Material.EmissiveColor", maMaterialEmissiveColor);
    mtr->SetUniform("u_Material.Shininess", maMaterialShininess);
    mtr->SetUniform("u_Material.DepthScale", maMaterialDepthScale);
    mtr->SetTexture("u_Material.DiffuseMap", m_material.diffuseMap);
    mtr->SetTexture("u_Material.NormalMap", m_material.normalMap);
    mtr->SetTexture("u_Material.SpecularMap", m_material.specularMap);
    mtr->SetTexture("u_Material.EmissiveMap", m_material.emissiveMap);
    mtr->SetTexture("u_Material.DepthMap", m_material.depthMap);

    mtr->SetUniform("u_Camera.PositionWS", maCameraPosition);
    // AmbientColor
    mtr->SetUniform("u_AmbientColor", maAmbientColor);

    mtr->SetUniformBuffer("Transform", Renderer::Resources::Get<UniformBuffer>("Transform"));
    mtr->SetUniformBuffer("Light", Renderer::Resources::Get<UniformBuffer>("Light"));

    Renderer::Resources::Create<Shader>("Blinn-Phong-Instance")->Define("INSTANCE|DIFFUSE_MAP|NORMAL_MAP")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");
    m_shaderBlinnPhong = Renderer::Resources::Create<Shader>("Blinn-Phong")->Define("DIFFUSE_MAP|NORMAL_MAP")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");
    Renderer::Resources::Create<Shader>("BlinnWithDiffuseNormalMap")->Define("DIFFUSE_MAP|NORMAL_MAP|DEPTH_MAP")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");
    Renderer::Resources::Create<Shader>("BlinnWithDiffuseMap")->Define("DIFFUSE_MAP")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");

    Renderer::Resources::Create<Renderer::Element>("UnitCubic")->Set(mesh, mtr);


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
    Renderer::Resources::Get<UniformBuffer>("Transform")->Upload("WS2CS", glm::value_ptr(m_viewport->GetCamera()->World2Clip()));
    m_fLight.pos = glm::vec4(cam->GetPosition(), 1);
    m_fLight.dir = glm::vec4(cam->GetDirection(), 0);
    Renderer::Resources::Get<UniformBuffer>("Light")->Upload("FlashLight", &m_fLight);
}

void LearnOpenGLLayer::_PrepareUniformBuffers()
{
    std::shared_ptr<UniformBuffer> ubTransform = Renderer::Resources::Create<UniformBuffer>("Transform")->SetSize(64);
    ubTransform->Push("WS2CS", glm::vec2(0, 64));
    ubTransform->Upload("WS2CS", glm::value_ptr(m_viewport->GetCamera()->World2Clip()));

    std::shared_ptr<UniformBuffer> ubLight = Renderer::Resources::Create<UniformBuffer>("Light")->SetSize(240);
    ubLight->Push("DirectionalLight", glm::vec2(0, 32));
    ubLight->Push("PointLight", glm::vec2(32, 48));
    ubLight->Push("SpotLight", glm::vec2(80, 64));
    ubLight->Push("FlashLight", glm::vec2(144, 64));
    ubLight->Upload("DirectionalLight", &m_dLight);
    ubLight->Upload("PointLight", &m_pLight);
    ubLight->Upload("SpotLight", &m_sLight);
    ubLight->Upload("FlashLight", &m_fLight);
}
