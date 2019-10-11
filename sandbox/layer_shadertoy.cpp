/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : layer_shadertoy.cpp
* author      : Garra
* time        : 2019-10-10 14:26:49
* description : 
*
============================================*/


#include "layer_shadertoy.h"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<ShaderToyLayer> ShaderToyLayer::Create()
{
    return std::make_shared<ShaderToyLayer>();
}

ShaderToyLayer::ShaderToyLayer()
    : Layer("ShaderToyLayer")
{
    _PrepareResources();
}

void ShaderToyLayer::_PrepareResources()
{
    unsigned char indices[] = { 0, 1, 2, 0, 2, 3 };
    Buffer::Layout layoutIndex = { { Buffer::Element::DataType::UChar } };
    std::shared_ptr<Buffer> indexBuffer = Buffer::CreateIndex(sizeof(indices), indices);
    indexBuffer->SetLayout(layoutIndex);


    float vertices[4*3] = 
    {
        -1.0f, -1.0f, 0.0f,
        +1.0f, -1.0f, 0.0f,
        +1.0f, +1.0f, 0.0f,
        -1.0f, +1.0f, 0.0f
    };
    Buffer::Layout layoutVertex = { { Buffer::Element::DataType::Float3, "a_Position", false }, };
    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(sizeof(vertices), vertices);
    vertexBuffer->SetLayout(layoutVertex);

    using MA = Material::Attribute;
    Renderer::Resources::Create<Mesh>("Canvas")->Set(indexBuffer, {vertexBuffer});
    Renderer::Resources::Create<Shader>("FlatColor")->LoadFromFile("/home/garra/study/dnn/assets/shader/FlatColor.glsl");
    Renderer::Resources::Create<Shader>("Shapes")->LoadFromFile("/home/garra/study/dnn/assets/shader/Shapes.glsl");
    Renderer::Resources::Create<Shader>("ShapingFunctions")->LoadFromFile("/home/garra/study/dnn/assets/shader/ShapingFunctions.glsl");
    std::shared_ptr<MA> maFlatColor = Renderer::Resources::Create<MA>("FlatColor")->Set(MA::Type::Float4, glm::value_ptr(glm::vec4(0.8, 0.1, 0.2, 1.0)));
    std::shared_ptr<MA> maResolution = Renderer::Resources::Create<MA>("Resolution")->Set(MA::Type::Float2, glm::value_ptr(glm::vec2(1000, 1000)));
    std::shared_ptr<MA> maShapingFunction = Renderer::Resources::Create<MA>("ShapingFunction")->Set(MA::Type::Int1, (void*)&m_shapingFunction);

//     Renderer::Resources::Create<Material>("Resolution")->Set("u_Resolution", maResolution);
//     Renderer::Resources::Create<Material>("FlatColor")->Set("u_Color", maFlatColor);
    Renderer::Resources::Create<Material>("Canvas")->Set("u_Color", maFlatColor)->Set("u_Resolution", maResolution)->Set("u_ShapingFunction", maShapingFunction);

    m_canvas = Renderer::Resources::Create<Renderer::Element>("Canvas")->Set(Renderer::Resources::Get<Mesh>("Canvas"), Renderer::Resources::Get<Material>("Canvas"));
    m_currentShader = Renderer::Resources::Get<Shader>("ShapingFunctions");
    m_testShader = TestShader::ShapingFunctions;
}

void ShaderToyLayer::OnUpdate(float deltaTime)
{
    Renderer::BeginScene(m_cameraController->GetCamera());
    Renderer::Submit(m_canvas, m_currentShader);
    Renderer::EndScene();
}

void ShaderToyLayer::OnEvent(Event& e)
{

}

void ShaderToyLayer::OnImGuiRender()
{
    ImGui::Begin("ShaderToyLayer");
    if(ImGui::RadioButton("FlatColor", m_testShader == TestShader::FlatColor))
    {
        m_testShader = TestShader::FlatColor;
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("ShapingFunctions", m_testShader == TestShader::ShapingFunctions))
    {
        m_testShader = TestShader::ShapingFunctions;
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("Shapes", m_testShader == TestShader::Shapes))
    {
        m_testShader = TestShader::Shapes;
    }

    switch(m_testShader)
    {
        case TestShader::FlatColor:
        {
            m_currentShader = Renderer::Resources::Get<Shader>("FlatColor");
            ImGui::ColorPicker4("Color", (float*)Renderer::Resources::Get<Material::Attribute>("FlatColor")->GetData());
            break;
        }
        case TestShader::Shapes:
        {
            m_currentShader = Renderer::Resources::Get<Shader>("Shapes");

            break;
        }
        case TestShader::ShapingFunctions:
        {
            m_currentShader = Renderer::Resources::Get<Shader>("ShapingFunctions");
#define RadioButtonOfFunction(name) \
            if(ImGui::RadioButton(#name, m_shapingFunction == ShapingFunction::name)) \
            {                                                                          \
                m_shapingFunction = ShapingFunction::name;                             \
            }                                                                          \

            RadioButtonOfFunction(Linear)
            RadioButtonOfFunction(Step)
            RadioButtonOfFunction(SmoothStep)
            RadioButtonOfFunction(Power)
            RadioButtonOfFunction(Sine)
            RadioButtonOfFunction(Cosine)
            RadioButtonOfFunction(BlinnWyvillCosineApproximation)
            RadioButtonOfFunction(DoubleCubicSeat)
            RadioButtonOfFunction(DoubleCubicSeatWidthLinearBlend)
            RadioButtonOfFunction(DoubleOddPolynomialSeat)
            RadioButtonOfFunction(SymmetricDoublePolynomialSigmoids)
            RadioButtonOfFunction(QuadraticThroughGivenPoint)
            RadioButtonOfFunction(ExponentialEaseIn)
            RadioButtonOfFunction(ExponentialEaseOut)
            RadioButtonOfFunction(ExponentialEasing)
            RadioButtonOfFunction(DoubleExponentialSeat)
            RadioButtonOfFunction(DoubleExponentialSigmoid)
            RadioButtonOfFunction(LogisticSigmoid)
            RadioButtonOfFunction(CircularEaseIn)
            RadioButtonOfFunction(CircularEaseOut)
            RadioButtonOfFunction(DoubleCircelSeat)
            RadioButtonOfFunction(DoubleCircleSigmoid)
            RadioButtonOfFunction(DoubleEllipticSeat)
            RadioButtonOfFunction(DoubleEllipticSigmoid)
            RadioButtonOfFunction(DoubleLinearWidthCircularFillet)
            RadioButtonOfFunction(CircularArcThroughGivenPoint)
            RadioButtonOfFunction(QuadraticBezier)
            RadioButtonOfFunction(CubicBezier)
            RadioButtonOfFunction(CubicBezierThroughTwoGivenPoints)
            RadioButtonOfFunction(Impulse)
            RadioButtonOfFunction(CubicPulse)
            RadioButtonOfFunction(ExponentialStep)
            RadioButtonOfFunction(Parabola)
            RadioButtonOfFunction(PowerCurve)
            Renderer::Resources::Get<Material::Attribute>("ShapingFunction")->UpdateData((void*)&m_shapingFunction);
        }
        default: break;
    }

    ImGui::End();
}
