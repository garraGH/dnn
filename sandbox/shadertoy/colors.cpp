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
    using MA = Material::Attribute;
    std::shared_ptr<MA> maColorFst = Renderer::Resources::Create<MA>("ColorFst")->Set(MA::Type::Float4, glm::value_ptr(glm::vec4(0.8, 0.1, 0.2, 1.0)));
    std::shared_ptr<MA> maColorSnd = Renderer::Resources::Create<MA>("ColorSnd")->Set(MA::Type::Float4, glm::value_ptr(glm::vec4(0.1, 0.8, 0.8, 1.0)));
    std::shared_ptr<MA> maTime = Renderer::Resources::Create<MA>("Time")->SetType(MA::Type::Float1);
    std::shared_ptr<MA> maSpeed = Renderer::Resources::Create<MA>("Speed")->SetType(MA::Type::Float1);
    std::shared_ptr<MA> maTest = Renderer::Resources::Create<MA>("Test")->SetType(MA::Type::Int1);
    std::shared_ptr<MA> maEasingFunction = Renderer::Resources::Create<MA>("EasingFunction")->SetType(MA::Type::Int1);
    std::shared_ptr<MA> maResolution = Renderer::Resources::Create<MA>("Resolution")->Set(MA::Type::Float2, glm::value_ptr(glm::vec2(1000, 1000)));
    std::shared_ptr<MA> maLineWidth = Renderer::Resources::Create<MA>("LineWidth")->SetType(MA::Type::Float1);
    std::shared_ptr<MA> maNumFlags = Renderer::Resources::Create<MA>("NumFlags")->SetType(MA::Type::Int1);
    std::shared_ptr<MA> maFlagLayout = Renderer::Resources::Create<MA>("FlagLayout")->SetType(MA::Type::Int1);
    std::shared_ptr<MA> maDirection = Renderer::Resources::Create<MA>("Direction")->SetType(MA::Type::Int1);
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("Colors");
    std::shared_ptr<Texture> texFst = Renderer::Resources::Create<Texture2D>("TexFst")->LoadFromFile("/home/garra/study/dnn/assets/texture/turner_sunset.jpg");
    std::shared_ptr<Texture> texSnd = Renderer::Resources::Create<Texture2D>("TexSnd")->LoadFromFile("/home/garra/study/dnn/assets/texture/noza.png");
    mtr->Set("u_ColorFst", maColorFst);
    mtr->Set("u_ColorSnd", maColorSnd);
    mtr->Set("u_Resolution", maResolution);
    mtr->Set("u_Time", maTime);
    mtr->Set("u_Speed", maSpeed);
    mtr->Set("u_EasingFunction", maEasingFunction);
    mtr->Set("u_Test", maTest);
    mtr->Set("u_LineWidth", maLineWidth);
    mtr->Set("u_NumFlags", maNumFlags);
    mtr->Set("u_FlagLayout", maFlagLayout);
    mtr->Set("u_Direction", maDirection);
    mtr->AddTexture("u_Texture2D_fst", texFst);
    mtr->AddTexture("u_Texture2D_snd", texSnd);

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

void Colors::OnUpdate()
{
    float t = m_timer->GetElapsedTime();
    Renderer::Resources::Get<Material::Attribute>("Time")->UpdateData((void*)&t);
}


void Colors::OnImGuiRender()
{
#define RadioButton(x)                               \
    if(ImGui::RadioButton(#x, *m_test == Test::x))   \
    {                                                \
        *m_test = Test::x;                           \
    }

    RadioButton(ColorTransition)
    RadioButton(TextureTransition)
    RadioButton(TextureMixEachChannel)
    RadioButton(ColorFlagsAnimation)
    RadioButton(ColorFlagsShrink)

#undef RadioButton


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
}

void Colors::_CreateColorTransitionGUI()
{
    ImGui::PushItemWidth(200);
    ImGui::ColorPicker4("ColorFst", (float*)Renderer::Resources::Get<Material::Attribute>("ColorFst")->GetData());
    ImGui::SameLine();
    ImGui::ColorPicker4("ColorSnd", (float*)Renderer::Resources::Get<Material::Attribute>("ColorSnd")->GetData());
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
