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
    std::shared_ptr<Mesh> msCanvas = Renderer::Resources::Create<Mesh>("Canvas")->Set(indexBuffer, {vertexBuffer});
    Renderer::Resources::Create<Shader>("FlatColor")->LoadFromFile("/home/garra/study/dnn/assets/shader/FlatColor.glsl");
    Renderer::Resources::Create<Shader>("Shapes")->LoadFromFile("/home/garra/study/dnn/assets/shader/Shapes.glsl");
    Renderer::Resources::Create<Shader>("ShapingFunctions")->LoadFromFile("/home/garra/study/dnn/assets/shader/ShapingFunctions.glsl");
    std::shared_ptr<MA> maFlatColor = Renderer::Resources::Create<MA>("FlatColor")->Set(MA::Type::Float4, glm::value_ptr(glm::vec4(0.8, 0.1, 0.2, 1.0)));
    std::shared_ptr<MA> maResolution = Renderer::Resources::Create<MA>("Resolution")->Set(MA::Type::Float2, glm::value_ptr(glm::vec2(1000, 1000)));
    std::shared_ptr<MA> maCoefficient = Renderer::Resources::Create<MA>("Coefficient")->Set(MA::Type::Float4, glm::value_ptr(glm::vec4(1, 0, 0, 0)));
    int data_initializer = 0;
    std::shared_ptr<MA> maShapingFunction = Renderer::Resources::Create<MA>("ShapingFunction")->Set(MA::Type::Int1, (void*)&data_initializer);
    m_shapingFunction = (ShapingFunction*)maShapingFunction->GetData();

    data_initializer = 1;
    std::shared_ptr<MA> maOrder = Renderer::Resources::Create<MA>("Order")->Set(MA::Type::Int1, (void*)&data_initializer);

//     Renderer::Resources::Create<Material>("Resolution")->Set("u_Resolution", maResolution);
//     Renderer::Resources::Create<Material>("FlatColor")->Set("u_Color", maFlatColor);
    std::shared_ptr<Material> mtrCanava = Renderer::Resources::Create<Material>("Canvas");
    mtrCanava->Set("u_Color", maFlatColor);
    mtrCanava->Set("u_Resolution", maResolution);
    mtrCanava->Set("u_ShapingFunction", maShapingFunction);
    mtrCanava->Set("u_Coefficient", maCoefficient);
    mtrCanava->Set("u_Order", maOrder);

    m_canvas = Renderer::Resources::Create<Renderer::Element>("Canvas")->Set(msCanvas, mtrCanava);
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
            float* param = (float*)Renderer::Resources::Get<Material::Attribute>("Coefficient")->GetData();
            int* order = (int*)Renderer::Resources::Get<Material::Attribute>("Order")->GetData();


            _ShapingFunction_Linear(param);
            _ShapingFunction_Step(param);
            _ShapingFunction_SmoothStep(param);
            _ShapingFunction_Power(param);
            _ShapingFunction_Sine(param);
            _ShapingFunction_Cosine(param);
            _ShapingFunction_BlinnWyvillCosineApproximation(param);
            _ShapingFunction_DoubleCubicSeat(param);
            _ShapingFunction_DoubleCubicSeatWidthLinearBlend(param);
            _ShapingFunction_DoubleOddPolynomialSeat(param, order);
            _ShapingFunction_SymmetricDoublePolynomialSigmoids(param, order);
            _ShapingFunction_QuadraticThroughGivenPoint(param);
            _ShapingFunction_ExponentialEaseIn(param);
            _ShapingFunction_ExponentialEaseOut(param);
            _ShapingFunction_ExponentialEasing(param);
            _ShapingFunction_DoubleExponentialSeat(param);
            _ShapingFunction_DoubleExponentialSigmoid(param);
            _ShapingFunction_LogisticSigmoid(param);
            _ShapingFunction_CircularEaseIn(param);
            _ShapingFunction_CircularEaseOut(param);
            _ShapingFunction_DoubleCircleSeat(param);
            _ShapingFunction_DoubleCircleSigmoid(param);
            _ShapingFunction_DoubleEllipticSeat(param);
            _ShapingFunction_DoubleEllipticSigmoid(param);
            _ShapingFunction_DoubleLinearWidthCircularFillet(param);
            _ShapingFunction_CircularArcThroughGivenPoint(param);
            _ShapingFunction_QuadraticBezier(param);
            _ShapingFunction_CubicBezier(param);
            _ShapingFunction_CubicBezierThroughTwoGivenPoints(param);
            _ShapingFunction_Impulse(param);
            _ShapingFunction_CubicPulse(param);
            _ShapingFunction_ExponentialStep(param);
            _ShapingFunction_Parabola(param);
            _ShapingFunction_PowerCurve(param);
        }
        default: break;
    }

    ImGui::End();
}


void ShaderToyLayer::_ShapingFunction_Linear(float* param)
{
    if(ImGui::RadioButton("Linear", *m_shapingFunction == ShapingFunction::Linear)) 
    {                                                                         
        *m_shapingFunction = ShapingFunction::Linear;                            
        param[0] = 1;
        param[1] = 0;
    }                                                                         
    if(*m_shapingFunction == ShapingFunction::Linear)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, -1, 1);
        ImGui::SameLine();
        ImGui::SliderFloat("b", param+1, -1, 1);
    }
}

void ShaderToyLayer::_ShapingFunction_Step(float* param)
{
    if(ImGui::RadioButton("Step", *m_shapingFunction == ShapingFunction::Step))
    {
        *m_shapingFunction = ShapingFunction::Step;
        param[0] = 0.5;
    }

    if(*m_shapingFunction == ShapingFunction::Step)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("threshold", param, 0, 1);
    }
}

void ShaderToyLayer::_ShapingFunction_SmoothStep(float* param)
{
    if(ImGui::RadioButton("SmoothStep", *m_shapingFunction == ShapingFunction::SmoothStep))
    {
        *m_shapingFunction = ShapingFunction::SmoothStep;
        param[0] = 0.2;
        param[1] = 0.6;
    }

    if(*m_shapingFunction == ShapingFunction::SmoothStep)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat("beg", param, 0, 1);
        ImGui::SameLine();
        ImGui::SliderFloat("end", param+1, 0, 1);
    }
}

void ShaderToyLayer::_ShapingFunction_Power(float* param)
{
    if(ImGui::RadioButton("Power", *m_shapingFunction == ShapingFunction::Power))
    {
        *m_shapingFunction = ShapingFunction::Power;
        param[0] = 2;
    }

    if(*m_shapingFunction == ShapingFunction::Power)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("pow", param, 0, 10);
    }
}

void ShaderToyLayer::_ShapingFunction_Sine(float* param)
{
    if(ImGui::RadioButton("Sine", *m_shapingFunction == ShapingFunction::Sine))
    {
        *m_shapingFunction = ShapingFunction::Sine;
    }
}

void ShaderToyLayer::_ShapingFunction_Cosine(float* param)
{
    if(ImGui::RadioButton("Cosine", *m_shapingFunction == ShapingFunction::Cosine))
    {
        *m_shapingFunction = ShapingFunction::Cosine;
    }
}


void ShaderToyLayer::_ShapingFunction_BlinnWyvillCosineApproximation(float* param)
{
    if(ImGui::RadioButton("BlinnWyvillCosineApproximation", *m_shapingFunction == ShapingFunction::BlinnWyvillCosineApproximation))
    {
        *m_shapingFunction = ShapingFunction::BlinnWyvillCosineApproximation;
    }
}

void ShaderToyLayer::_ShapingFunction_DoubleCubicSeat(float* param)
{
    if(ImGui::RadioButton("DoubleCubicSeat", *m_shapingFunction == ShapingFunction::DoubleCubicSeat))
    {
        *m_shapingFunction = ShapingFunction::DoubleCubicSeat;
        param[0] = 0.3;
        param[1] = 0.5;
    }

    if(*m_shapingFunction == ShapingFunction::DoubleCubicSeat)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_DoubleCubicSeatWidthLinearBlend(float* param)
{
    if(ImGui::RadioButton("DoubleCubicSeatWidthLinearBlend", *m_shapingFunction == ShapingFunction::DoubleCubicSeatWidthLinearBlend))
    {
        *m_shapingFunction = ShapingFunction::DoubleCubicSeatWidthLinearBlend;
        param[0] = 0.3;
        param[1] = 0.5;
    }

    if(*m_shapingFunction == ShapingFunction::DoubleCubicSeatWidthLinearBlend)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_DoubleOddPolynomialSeat(float* param, int* order)
{
    if(ImGui::RadioButton("DoubleOddPolynomialSeat", *m_shapingFunction == ShapingFunction::DoubleOddPolynomialSeat))
    {
        *m_shapingFunction = ShapingFunction::DoubleOddPolynomialSeat;
        param[0] = 0.3;
        param[1] = 0.5;
        *order = 1;
    }

    if(*m_shapingFunction == ShapingFunction::DoubleOddPolynomialSeat)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
        ImGui::SameLine();
        ImGui::SliderInt("order", order, -1, 10);
    } 
}

void ShaderToyLayer::_ShapingFunction_SymmetricDoublePolynomialSigmoids(float* param, int* order)
{
    if(ImGui::RadioButton("SymmetricDoublePolynomialSigmoids", *m_shapingFunction == ShapingFunction::SymmetricDoublePolynomialSigmoids))
    {
        *m_shapingFunction = ShapingFunction::SymmetricDoublePolynomialSigmoids;
        param[0] = 0.3;
        param[1] = 0.5;
        *order = 1;
    }

    if(*m_shapingFunction == ShapingFunction::SymmetricDoublePolynomialSigmoids)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderInt("order", order, -10, 10);
    } 
}

void ShaderToyLayer::_ShapingFunction_QuadraticThroughGivenPoint(float* param)
{
    if(ImGui::RadioButton("QuadraticThroughGivenPoint", *m_shapingFunction == ShapingFunction::QuadraticThroughGivenPoint))
    {
        *m_shapingFunction = ShapingFunction::QuadraticThroughGivenPoint;
        param[0] = 0.3;
        param[1] = 0.5;
    }

    if(*m_shapingFunction == ShapingFunction::QuadraticThroughGivenPoint)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_ExponentialEaseIn(float* param)
{
    if(ImGui::RadioButton("ExponentialEaseIn", *m_shapingFunction == ShapingFunction::ExponentialEaseIn))
    {
        *m_shapingFunction = ShapingFunction::ExponentialEaseIn;
        param[0] = 0.2;
    }

    if(*m_shapingFunction == ShapingFunction::ExponentialEaseIn)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_ExponentialEaseOut(float* param)
{
    if(ImGui::RadioButton("ExponentialEaseOut", *m_shapingFunction == ShapingFunction::ExponentialEaseOut))
    {
        *m_shapingFunction = ShapingFunction::ExponentialEaseOut;
        param[0] = 0.2;
    }

    if(*m_shapingFunction == ShapingFunction::ExponentialEaseOut)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_ExponentialEasing(float* param)
{
    if(ImGui::RadioButton("ExponentialEasing", *m_shapingFunction == ShapingFunction::ExponentialEasing))
    {
        *m_shapingFunction = ShapingFunction::ExponentialEasing;
        param[0] = 0.2;
    }

    if(*m_shapingFunction == ShapingFunction::ExponentialEasing)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_DoubleExponentialSeat(float* param)
{
    if(ImGui::RadioButton("DoubleExponentialSeat", *m_shapingFunction == ShapingFunction::DoubleExponentialSeat))
    {
        *m_shapingFunction = ShapingFunction::DoubleExponentialSeat;
        param[0] = 0.2;
    }

    if(*m_shapingFunction == ShapingFunction::DoubleExponentialSeat)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_DoubleExponentialSigmoid(float* param)
{
    if(ImGui::RadioButton("DoubleExponentialSigmoid", *m_shapingFunction == ShapingFunction::DoubleExponentialSigmoid))
    {
        *m_shapingFunction = ShapingFunction::DoubleExponentialSigmoid;
        param[0] = 0.2;
    }

    if(*m_shapingFunction == ShapingFunction::DoubleExponentialSigmoid)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_LogisticSigmoid(float* param)
{
    if(ImGui::RadioButton("LogisticSigmoid", *m_shapingFunction == ShapingFunction::LogisticSigmoid))
    {
        *m_shapingFunction = ShapingFunction::LogisticSigmoid;
        param[0] = 0.8;
    }

    if(*m_shapingFunction == ShapingFunction::LogisticSigmoid)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_CircularEaseIn(float* param)
{
    if(ImGui::RadioButton("CircularEaseIn", *m_shapingFunction == ShapingFunction::CircularEaseIn))
    {
        *m_shapingFunction = ShapingFunction::CircularEaseIn;
    }
}

void ShaderToyLayer::_ShapingFunction_CircularEaseOut(float* param)
{
    if(ImGui::RadioButton("CircularEaseOut", *m_shapingFunction == ShapingFunction::CircularEaseOut))
    {
        *m_shapingFunction = ShapingFunction::CircularEaseOut;
    }
}

void ShaderToyLayer::_ShapingFunction_DoubleCircleSeat(float* param)
{
    if(ImGui::RadioButton("DoubleCircleSeat", *m_shapingFunction == ShapingFunction::DoubleCircleSeat))
    {
        *m_shapingFunction = ShapingFunction::DoubleCircleSeat;
        param[0] = 0.2;
    }

    if(*m_shapingFunction == ShapingFunction::DoubleCircleSeat)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_DoubleCircleSigmoid(float* param)
{
    if(ImGui::RadioButton("DoubleCircleSigmoid", *m_shapingFunction == ShapingFunction::DoubleCircleSigmoid))
    {
        *m_shapingFunction = ShapingFunction::DoubleCircleSigmoid;
        param[0] = 0.2;
    }

    if(*m_shapingFunction == ShapingFunction::DoubleCircleSigmoid)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_DoubleEllipticSeat(float* param)
{
    if(ImGui::RadioButton("DoubleEllipticSeat", *m_shapingFunction == ShapingFunction::DoubleEllipticSeat))
    {
        *m_shapingFunction = ShapingFunction::DoubleEllipticSeat;
        param[0] = 0.2;
        param[1] = 0.5;
    }

    if(*m_shapingFunction == ShapingFunction::DoubleEllipticSeat)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_DoubleEllipticSigmoid(float* param)
{
    if(ImGui::RadioButton("DoubleEllipticSigmoid", *m_shapingFunction == ShapingFunction::DoubleEllipticSigmoid))
    {
        *m_shapingFunction = ShapingFunction::DoubleEllipticSigmoid;
        param[0] = 0.2;
        param[1] = 0.5;
    }

    if(*m_shapingFunction == ShapingFunction::DoubleEllipticSigmoid)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_DoubleLinearWidthCircularFillet(float* param)
{
    if(ImGui::RadioButton("DoubleLinearWidthCircularFillet", *m_shapingFunction == ShapingFunction::DoubleLinearWidthCircularFillet))
    {
        *m_shapingFunction = ShapingFunction::DoubleLinearWidthCircularFillet;
        param[0] = 0.713;
        param[1] = 0.267;
        param[2] = 0.25;
    }

    if(*m_shapingFunction == ShapingFunction::DoubleLinearWidthCircularFillet)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
        ImGui::SameLine();
        ImGui::SliderFloat("radius", param+2, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_CircularArcThroughGivenPoint(float* param)
{
    if(ImGui::RadioButton("CircularArcThroughGivenPoint", *m_shapingFunction == ShapingFunction::CircularArcThroughGivenPoint))
    {
        *m_shapingFunction = ShapingFunction::CircularArcThroughGivenPoint;
        param[0] = 0.553;
        param[1] = 0.840;
    }

    if(*m_shapingFunction == ShapingFunction::CircularArcThroughGivenPoint)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_QuadraticBezier(float* param)
{
    if(ImGui::RadioButton("QuadraticBezier", *m_shapingFunction == ShapingFunction::QuadraticBezier))
    {
        *m_shapingFunction = ShapingFunction::QuadraticBezier;
        param[0] = 0.94;
        param[1] = 0.28;
    }

    if(*m_shapingFunction == ShapingFunction::QuadraticBezier)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_CubicBezier(float* param)
{
    if(ImGui::RadioButton("CubicBezier", *m_shapingFunction == ShapingFunction::CubicBezier))
    {
        *m_shapingFunction = ShapingFunction::CubicBezier;
        param[0] = 0.253;
        param[1] = 0.72;
        param[2] = 0.753;
        param[3] = 0.253;
    }

    if(*m_shapingFunction == ShapingFunction::CubicBezier)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat4("pnt", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_CubicBezierThroughTwoGivenPoints(float* param)
{
    if(ImGui::RadioButton("CubicBezierThroughTwoGivenPoints", *m_shapingFunction == ShapingFunction::CubicBezierThroughTwoGivenPoints))
    {
        *m_shapingFunction = ShapingFunction::CubicBezierThroughTwoGivenPoints;
        param[0] = 0.293;
        param[1] = 0.667;
        param[2] = 0.75;
        param[3] = 0.25;
    }

    if(*m_shapingFunction == ShapingFunction::CubicBezierThroughTwoGivenPoints)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat4("pnt", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_Impulse(float* param)
{
    if(ImGui::RadioButton("Impulse", *m_shapingFunction == ShapingFunction::Impulse))
    {
        *m_shapingFunction = ShapingFunction::Impulse;
        param[0] = 2;
    }

    if(*m_shapingFunction == ShapingFunction::Impulse)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("order", param, 0, 20);
    } 
}

void ShaderToyLayer::_ShapingFunction_CubicPulse(float* param)
{
    if(ImGui::RadioButton("CubicPulse", *m_shapingFunction == ShapingFunction::CubicPulse))
    {
        *m_shapingFunction = ShapingFunction::CubicPulse;
        param[0] = 0.5;
        param[1] = 0.1;
    }

    if(*m_shapingFunction == ShapingFunction::CubicPulse)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_ExponentialStep(float* param)
{
    if(ImGui::RadioButton("ExponentialStep", *m_shapingFunction == ShapingFunction::ExponentialStep))
    {
        *m_shapingFunction = ShapingFunction::ExponentialStep;
        param[0] = 10;
        param[1] = 1;
    }

    if(*m_shapingFunction == ShapingFunction::ExponentialStep)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat("order", param, 0, 10);
        ImGui::SameLine();
        ImGui::SliderFloat("threshold", param+1, 0, 1);
    } 
}

void ShaderToyLayer::_ShapingFunction_Parabola(float* param)
{
    if(ImGui::RadioButton("Parabola", *m_shapingFunction == ShapingFunction::Parabola))
    {
        *m_shapingFunction = ShapingFunction::Parabola;
        param[0] = 1;
    }

    if(*m_shapingFunction == ShapingFunction::Parabola)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("order", param, -20, 20);
    } 
}

void ShaderToyLayer::_ShapingFunction_PowerCurve(float* param)
{
    if(ImGui::RadioButton("PowerCurve", *m_shapingFunction == ShapingFunction::PowerCurve))
    {
        *m_shapingFunction = ShapingFunction::PowerCurve;
        param[0] = 3;
        param[1] = 1;
    }

    if(*m_shapingFunction == ShapingFunction::PowerCurve)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 10);
    } 
}
