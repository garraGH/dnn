/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : colors.cpp
* author      : Garra
* time        : 2019-10-13 09:50:21
* description : 
*
============================================*/


#include "colors.h"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<Colors> Colors::Create()
{
    return std::make_shared<Colors>();
}

Colors::Colors()
{
    m_type = Type::Colors;
    _PrepareResources();
}


void Colors::_PrepareResources()
{
    using MU = Material::Uniform;
    std::shared_ptr<MU> maColorFst = Renderer::Resources::Create<MU>("ColorFst")->Set(MU::Type::Float4, 1, glm::value_ptr(glm::vec4(0.8, 0.1, 0.2, 1.0)));
    std::shared_ptr<MU> maColorSnd = Renderer::Resources::Create<MU>("ColorSnd")->Set(MU::Type::Float4, 1, glm::value_ptr(glm::vec4(0.1, 0.8, 0.8, 1.0)));
    std::shared_ptr<MU> maTime = Renderer::Resources::Create<MU>("Time")->SetType(MU::Type::Float1);
    std::shared_ptr<MU> maSpeed = Renderer::Resources::Create<MU>("Speed")->SetType(MU::Type::Float1);
    std::shared_ptr<MU> maTest = Renderer::Resources::Create<MU>("Test")->SetType(MU::Type::Int1);
    std::shared_ptr<MU> maEasingFunction = Renderer::Resources::Create<MU>("EasingFunction")->SetType(MU::Type::Int1);
    std::shared_ptr<MU> maResolution = Renderer::Resources::Create<MU>("Resolution")->Set(MU::Type::Float2, 1, glm::value_ptr(glm::vec2(1000, 1000)));
    std::shared_ptr<MU> maLineWidth = Renderer::Resources::Create<MU>("LineWidth")->SetType(MU::Type::Float1);
    std::shared_ptr<MU> maNumFlags = Renderer::Resources::Create<MU>("NumFlags")->SetType(MU::Type::Int1);
    std::shared_ptr<MU> maFlagLayout = Renderer::Resources::Create<MU>("FlagLayout")->SetType(MU::Type::Int1);
    std::shared_ptr<MU> maDirection = Renderer::Resources::Create<MU>("Direction")->SetType(MU::Type::Int1);
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("Colors");
    std::shared_ptr<Texture> texFst = Renderer::Resources::Create<Texture2D>("TexFst")->LoadFromFile("/home/garra/study/dnn/assets/texture/turner_sunset.jpg");
    std::shared_ptr<Texture> texSnd = Renderer::Resources::Create<Texture2D>("TexSnd")->LoadFromFile("/home/garra/study/dnn/assets/texture/noza.png");
    mtr->SetUniform("u_ColorFst", maColorFst);
    mtr->SetUniform("u_ColorSnd", maColorSnd);
    mtr->SetUniform("u_Resolution", maResolution);
    mtr->SetUniform("u_Time", maTime);
    mtr->SetUniform("u_Speed", maSpeed);
    mtr->SetUniform("u_EasingFunction", maEasingFunction);
    mtr->SetUniform("u_Test", maTest);
    mtr->SetUniform("u_LineWidth", maLineWidth);
    mtr->SetUniform("u_NumFlags", maNumFlags);
    mtr->SetUniform("u_FlagLayout", maFlagLayout);
    mtr->SetUniform("u_Direction", maDirection);
    mtr->SetTexture("u_Texture2D_fst", texFst);
    mtr->SetTexture("u_Texture2D_snd", texSnd);

    m_speed = (float*)maSpeed->GetData();
    *m_speed = 1.0;

    m_lineWidth = (float*)maLineWidth->GetData();
    *m_lineWidth = 4.0;

    m_test = (Test*)maTest->GetData();
    *m_test = Test::ColorTransition;

    m_easingFunction = (EasingFunction*)maEasingFunction->GetData();
    *m_easingFunction = EasingFunction::Linear;

    m_nFlags = (int*)maNumFlags->GetData();
    *m_nFlags = 10;

    m_flagLayout = (FlagLayout*)maFlagLayout->GetData();
    *m_flagLayout = FlagLayout::Vertical;

    m_direction = (Direction*)maDirection->GetData();
    *m_direction = Direction::CounterClockwise;

    Renderer::Resources::Create<Shader>("Colors")->LoadFromFile("/home/garra/study/dnn/assets/shader/Colors.glsl");
}


std::shared_ptr<Material> Colors::GetMaterial() const
{
    return Renderer::Resources::Get<Material>("Colors");
}

std::shared_ptr<Shader> Colors::GetShader() const
{
    return Renderer::Resources::Get<Shader>("Colors");
}

void Colors::OnUpdate(float deltaTime)
{
    float t = m_timer->GetElapsedTime();
    Renderer::Resources::Get<Material::Uniform>("Time")->UpdateData((void*)&t);
}


void Colors::OnImGuiRender()
{
#define RadioButton(x)                               \
    if(ImGui::RadioButton(#x, *m_test == Test::x))   \
    {                                                \
        *m_test = Test::x;                           \
    }


    RadioButton(ColorTransition)
    ImGui::SameLine();
    RadioButton(TextureTransition)
    ImGui::SameLine();
    RadioButton(TextureMixEachChannel)
    ImGui::SameLine();
    RadioButton(ColorFlagsAnimation)
    ImGui::SameLine();
    RadioButton(ColorFlagsShrink)

#undef RadioButton

    ImGui::Separator();

    switch(*m_test)
    {
        case Test::ColorTransition: 
            _CreateColorTransitionGUI();
            break;
        case Test::TextureTransition:
            _CreateTextureTransitionGUI();
            break;
        case Test::ColorFlagsAnimation:
            _CreateColorFlagsAnimationGUI();
            break;
        case Test::ColorFlagsShrink:
            _CreateColorFlagsShrink();
            break;
        default:
            _CreateTextureMixEachChannelGUI();
    }
    ImGui::Separator();
}

void Colors::_CreateColorTransitionGUI()
{
    ImGui::PushItemWidth(200);
    ImGui::ColorPicker4("ColorFst", (float*)Renderer::Resources::Get<Material::Uniform>("ColorFst")->GetData());
    ImGui::SameLine();
    ImGui::ColorPicker4("ColorSnd", (float*)Renderer::Resources::Get<Material::Uniform>("ColorSnd")->GetData());
    ImGui::SliderFloat("Speed", m_speed, 0, 3);

    _CreateEasingFunctionsGUI();
}

void Colors::_CreateTextureTransitionGUI()
{
    std::shared_ptr<Texture> texFst = Renderer::Resources::Get<Texture2D>("TexFst");
    std::shared_ptr<Texture> texSnd = Renderer::Resources::Get<Texture2D>("TexSnd");

    std::string fstImagePath = texFst->GetImagePath();
    std::string sndImagePath = texSnd->GetImagePath();

    char imagepath[256];
    int size = fstImagePath.size();
    memcpy(imagepath, fstImagePath.c_str(), size);
    imagepath[size] = '\0';
    if(ImGui::InputText("First  image path", imagepath, 256, ImGuiInputTextFlags_EnterReturnsTrue))
    {
        texFst->LoadFromFile(imagepath);
    }
    size = sndImagePath.size();
    memcpy(imagepath, sndImagePath.c_str(), size);
    imagepath[size] = '\0';
    if(ImGui::InputText("Second image path", imagepath, 256, ImGuiInputTextFlags_EnterReturnsTrue))
    {
        texSnd->LoadFromFile(imagepath);
    }
    ImGui::SliderFloat("Speed", m_speed, 0.1, 3);

    _CreateEasingFunctionsGUI();
}

void Colors::_CreateTextureMixEachChannelGUI()
{
    ImGui::SliderFloat("LineWidth", m_lineWidth, 1, 16);
}

void Colors::_CreateEasingFunctionsGUI()
{

#define RadioButton(x) \
    if(ImGui::RadioButton(#x, *m_easingFunction == EasingFunction::x))\
    {                                                                 \
        *m_easingFunction = EasingFunction::x;                        \
    }                                                                 \

    RadioButton(Linear)
    RadioButton(ExponentialIn)
    RadioButton(ExponentialOut)
    RadioButton(ExponentialInOut)
    RadioButton(SineIn)
    RadioButton(SineOut)
    RadioButton(SineInOut)
    RadioButton(QinticIn)
    RadioButton(QinticOut)
    RadioButton(QinticInOut)
    RadioButton(QuarticIn)
    RadioButton(QuarticOut)
    RadioButton(QuarticInOut)
    RadioButton(QuadraticIn)
    RadioButton(QuadraticOut)
    RadioButton(QuadraticInOut)
    RadioButton(CubicIn)
    RadioButton(CubicOut)
    RadioButton(CubicInOut)
    RadioButton(ElasticIn)
    RadioButton(ElasticOut)
    RadioButton(ElasticInOut)
    RadioButton(CircularIn)
    RadioButton(CircularOut)
    RadioButton(CircularInOut)
    RadioButton(BounceIn)
    RadioButton(BounceOut)
    RadioButton(BounceInOut)
    RadioButton(BackIn)
    RadioButton(BackOut)
    RadioButton(BackInOut)
#undef RadioButton
}

void Colors::_CreateColorFlagsAnimationGUI()
{
#define RadioButton(x) \
    if(ImGui::RadioButton(#x, *m_flagLayout == FlagLayout::x))\
    {                                                         \
        *m_flagLayout = FlagLayout::x;                        \
    }                                                         \
    
    RadioButton(Vertical)
    ImGui::SameLine();
    RadioButton(Horizontal)
    ImGui::SameLine();
    RadioButton(Circle)
    ImGui::SameLine();
    RadioButton(Rainbow)

#undef RadioButton

    ImGui::PushItemWidth(100);
    if(ImGui::RadioButton("Clockwise", *m_direction == Direction::Clockwise))
    {
        *m_direction = Direction::Clockwise;
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("CounterClockwise", *m_direction == Direction::CounterClockwise))
    {
        *m_direction = Direction::CounterClockwise;
    }
    ImGui::SameLine();
    ImGui::SliderInt("number of flags", m_nFlags, 1, 360);
    ImGui::SameLine();
    ImGui::SliderFloat("speed", m_speed, 0, 3);
}


void Colors::_CreateColorFlagsShrink()
{

}             
