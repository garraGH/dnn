/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : shaping_functions.cpp
* author      : Garra
* time        : 2019-10-13 09:50:21
* description : 
*
============================================*/


#include "shaping_functions.h"
#include "elsa.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<ShapingFunctions> ShapingFunctions::Create()
{
    return std::make_shared<ShapingFunctions>();
}

ShapingFunctions::ShapingFunctions()
{
    m_type = Type::ShapingFunctions;
    _PrepareResources();
}

void ShapingFunctions::_PrepareResources()
{

    using MA = Material::Attribute;
    Renderer::Resources::Create<Shader>("ShapingFunctions")->LoadFromFile("/home/garra/study/dnn/assets/shader/ShapingFunctions.glsl");
    std::shared_ptr<MA> maResolution = Renderer::Resources::Create<MA>("Resolution")->Set(MA::Type::Float2, 1, glm::value_ptr(glm::vec2(1000, 1000)));
    std::shared_ptr<MA> maCoefficient = Renderer::Resources::Create<MA>("Coefficient")->Set(MA::Type::Float4, 1, glm::value_ptr(glm::vec4(1, 0, 0, 0)));
    std::shared_ptr<MA> maFunction = Renderer::Resources::Create<MA>("Function")->SetType(MA::Type::Int1);
    std::shared_ptr<MA> maOrder = Renderer::Resources::Create<MA>("Order")->SetType(MA::Type::Int1);

    m_function = (Function*)maFunction->GetData();
    *m_function = Function::Linear;
    int* order = (int*)maOrder->GetData();
    *order = 1;

    std::shared_ptr<Material> mtrCanava = Renderer::Resources::Create<Material>("ShapingFunctions");
    mtrCanava->Set("u_Resolution", maResolution);
    mtrCanava->Set("u_Function", maFunction);
    mtrCanava->Set("u_Coefficient", maCoefficient);
    mtrCanava->Set("u_Order", maOrder);
}

std::shared_ptr<Material> ShapingFunctions::GetMaterial() const
{
    return Renderer::Resources::Get<Material>("ShapingFunctions");
}

std::shared_ptr<Shader> ShapingFunctions::GetShader() const
{
    return Renderer::Resources::Get<Shader>("ShapingFunctions");
}

void ShapingFunctions::OnImGuiRender()
{
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

void ShapingFunctions::_ShapingFunction_Linear(float* param)
{
    if(ImGui::RadioButton("Linear", *m_function == Function::Linear)) 
    {                                                                         
        *m_function = Function::Linear;                            
        param[0] = 1;
        param[1] = 0;
    }                                                                         
    if(*m_function == Function::Linear)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, -1, 1);
        ImGui::SameLine();
        ImGui::SliderFloat("b", param+1, -1, 1);
    }
}

void ShapingFunctions::_ShapingFunction_Step(float* param)
{
    if(ImGui::RadioButton("Step", *m_function == Function::Step))
    {
        *m_function = Function::Step;
        param[0] = 0.5;
    }

    if(*m_function == Function::Step)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("threshold", param, 0, 1);
    }
}

void ShapingFunctions::_ShapingFunction_SmoothStep(float* param)
{
    if(ImGui::RadioButton("SmoothStep", *m_function == Function::SmoothStep))
    {
        *m_function = Function::SmoothStep;
        param[0] = 0.2;
        param[1] = 0.6;
    }

    if(*m_function == Function::SmoothStep)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat("beg", param, 0, 1);
        ImGui::SameLine();
        ImGui::SliderFloat("end", param+1, 0, 1);
    }
}

void ShapingFunctions::_ShapingFunction_Power(float* param)
{
    if(ImGui::RadioButton("Power", *m_function == Function::Power))
    {
        *m_function = Function::Power;
        param[0] = 2;
    }

    if(*m_function == Function::Power)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("pow", param, 0, 10);
    }
}

void ShapingFunctions::_ShapingFunction_Sine(float* param)
{
    if(ImGui::RadioButton("Sine", *m_function == Function::Sine))
    {
        *m_function = Function::Sine;
    }
}

void ShapingFunctions::_ShapingFunction_Cosine(float* param)
{
    if(ImGui::RadioButton("Cosine", *m_function == Function::Cosine))
    {
        *m_function = Function::Cosine;
    }
}


void ShapingFunctions::_ShapingFunction_BlinnWyvillCosineApproximation(float* param)
{
    if(ImGui::RadioButton("BlinnWyvillCosineApproximation", *m_function == Function::BlinnWyvillCosineApproximation))
    {
        *m_function = Function::BlinnWyvillCosineApproximation;
    }
}

void ShapingFunctions::_ShapingFunction_DoubleCubicSeat(float* param)
{
    if(ImGui::RadioButton("DoubleCubicSeat", *m_function == Function::DoubleCubicSeat))
    {
        *m_function = Function::DoubleCubicSeat;
        param[0] = 0.3;
        param[1] = 0.5;
    }

    if(*m_function == Function::DoubleCubicSeat)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_DoubleCubicSeatWidthLinearBlend(float* param)
{
    if(ImGui::RadioButton("DoubleCubicSeatWidthLinearBlend", *m_function == Function::DoubleCubicSeatWidthLinearBlend))
    {
        *m_function = Function::DoubleCubicSeatWidthLinearBlend;
        param[0] = 0.3;
        param[1] = 0.5;
    }

    if(*m_function == Function::DoubleCubicSeatWidthLinearBlend)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_DoubleOddPolynomialSeat(float* param, int* order)
{
    if(ImGui::RadioButton("DoubleOddPolynomialSeat", *m_function == Function::DoubleOddPolynomialSeat))
    {
        *m_function = Function::DoubleOddPolynomialSeat;
        param[0] = 0.3;
        param[1] = 0.5;
        *order = 1;
    }

    if(*m_function == Function::DoubleOddPolynomialSeat)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
        ImGui::SameLine();
        ImGui::SliderInt("order", order, -1, 10);
    } 
}

void ShapingFunctions::_ShapingFunction_SymmetricDoublePolynomialSigmoids(float* param, int* order)
{
    if(ImGui::RadioButton("SymmetricDoublePolynomialSigmoids", *m_function == Function::SymmetricDoublePolynomialSigmoids))
    {
        *m_function = Function::SymmetricDoublePolynomialSigmoids;
        param[0] = 0.3;
        param[1] = 0.5;
        *order = 1;
    }

    if(*m_function == Function::SymmetricDoublePolynomialSigmoids)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderInt("order", order, -10, 10);
    } 
}

void ShapingFunctions::_ShapingFunction_QuadraticThroughGivenPoint(float* param)
{
    if(ImGui::RadioButton("QuadraticThroughGivenPoint", *m_function == Function::QuadraticThroughGivenPoint))
    {
        *m_function = Function::QuadraticThroughGivenPoint;
        param[0] = 0.3;
        param[1] = 0.5;
    }

    if(*m_function == Function::QuadraticThroughGivenPoint)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_ExponentialEaseIn(float* param)
{
    if(ImGui::RadioButton("ExponentialEaseIn", *m_function == Function::ExponentialEaseIn))
    {
        *m_function = Function::ExponentialEaseIn;
        param[0] = 0.2;
    }

    if(*m_function == Function::ExponentialEaseIn)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_ExponentialEaseOut(float* param)
{
    if(ImGui::RadioButton("ExponentialEaseOut", *m_function == Function::ExponentialEaseOut))
    {
        *m_function = Function::ExponentialEaseOut;
        param[0] = 0.2;
    }

    if(*m_function == Function::ExponentialEaseOut)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_ExponentialEasing(float* param)
{
    if(ImGui::RadioButton("ExponentialEasing", *m_function == Function::ExponentialEasing))
    {
        *m_function = Function::ExponentialEasing;
        param[0] = 0.2;
    }

    if(*m_function == Function::ExponentialEasing)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_DoubleExponentialSeat(float* param)
{
    if(ImGui::RadioButton("DoubleExponentialSeat", *m_function == Function::DoubleExponentialSeat))
    {
        *m_function = Function::DoubleExponentialSeat;
        param[0] = 0.2;
    }

    if(*m_function == Function::DoubleExponentialSeat)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_DoubleExponentialSigmoid(float* param)
{
    if(ImGui::RadioButton("DoubleExponentialSigmoid", *m_function == Function::DoubleExponentialSigmoid))
    {
        *m_function = Function::DoubleExponentialSigmoid;
        param[0] = 0.2;
    }

    if(*m_function == Function::DoubleExponentialSigmoid)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_LogisticSigmoid(float* param)
{
    if(ImGui::RadioButton("LogisticSigmoid", *m_function == Function::LogisticSigmoid))
    {
        *m_function = Function::LogisticSigmoid;
        param[0] = 0.8;
    }

    if(*m_function == Function::LogisticSigmoid)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_CircularEaseIn(float* param)
{
    if(ImGui::RadioButton("CircularEaseIn", *m_function == Function::CircularEaseIn))
    {
        *m_function = Function::CircularEaseIn;
    }
}

void ShapingFunctions::_ShapingFunction_CircularEaseOut(float* param)
{
    if(ImGui::RadioButton("CircularEaseOut", *m_function == Function::CircularEaseOut))
    {
        *m_function = Function::CircularEaseOut;
    }
}

void ShapingFunctions::_ShapingFunction_DoubleCircleSeat(float* param)
{
    if(ImGui::RadioButton("DoubleCircleSeat", *m_function == Function::DoubleCircleSeat))
    {
        *m_function = Function::DoubleCircleSeat;
        param[0] = 0.2;
    }

    if(*m_function == Function::DoubleCircleSeat)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_DoubleCircleSigmoid(float* param)
{
    if(ImGui::RadioButton("DoubleCircleSigmoid", *m_function == Function::DoubleCircleSigmoid))
    {
        *m_function = Function::DoubleCircleSigmoid;
        param[0] = 0.2;
    }

    if(*m_function == Function::DoubleCircleSigmoid)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("a", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_DoubleEllipticSeat(float* param)
{
    if(ImGui::RadioButton("DoubleEllipticSeat", *m_function == Function::DoubleEllipticSeat))
    {
        *m_function = Function::DoubleEllipticSeat;
        param[0] = 0.2;
        param[1] = 0.5;
    }

    if(*m_function == Function::DoubleEllipticSeat)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_DoubleEllipticSigmoid(float* param)
{
    if(ImGui::RadioButton("DoubleEllipticSigmoid", *m_function == Function::DoubleEllipticSigmoid))
    {
        *m_function = Function::DoubleEllipticSigmoid;
        param[0] = 0.2;
        param[1] = 0.5;
    }

    if(*m_function == Function::DoubleEllipticSigmoid)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_DoubleLinearWidthCircularFillet(float* param)
{
    if(ImGui::RadioButton("DoubleLinearWidthCircularFillet", *m_function == Function::DoubleLinearWidthCircularFillet))
    {
        *m_function = Function::DoubleLinearWidthCircularFillet;
        param[0] = 0.713;
        param[1] = 0.267;
        param[2] = 0.25;
    }

    if(*m_function == Function::DoubleLinearWidthCircularFillet)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
        ImGui::SameLine();
        ImGui::SliderFloat("radius", param+2, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_CircularArcThroughGivenPoint(float* param)
{
    if(ImGui::RadioButton("CircularArcThroughGivenPoint", *m_function == Function::CircularArcThroughGivenPoint))
    {
        *m_function = Function::CircularArcThroughGivenPoint;
        param[0] = 0.553;
        param[1] = 0.840;
    }

    if(*m_function == Function::CircularArcThroughGivenPoint)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_QuadraticBezier(float* param)
{
    if(ImGui::RadioButton("QuadraticBezier", *m_function == Function::QuadraticBezier))
    {
        *m_function = Function::QuadraticBezier;
        param[0] = 0.94;
        param[1] = 0.28;
    }

    if(*m_function == Function::QuadraticBezier)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_CubicBezier(float* param)
{
    if(ImGui::RadioButton("CubicBezier", *m_function == Function::CubicBezier))
    {
        *m_function = Function::CubicBezier;
        param[0] = 0.253;
        param[1] = 0.72;
        param[2] = 0.753;
        param[3] = 0.253;
    }

    if(*m_function == Function::CubicBezier)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat4("pnt", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_CubicBezierThroughTwoGivenPoints(float* param)
{
    if(ImGui::RadioButton("CubicBezierThroughTwoGivenPoints", *m_function == Function::CubicBezierThroughTwoGivenPoints))
    {
        *m_function = Function::CubicBezierThroughTwoGivenPoints;
        param[0] = 0.293;
        param[1] = 0.667;
        param[2] = 0.75;
        param[3] = 0.25;
    }

    if(*m_function == Function::CubicBezierThroughTwoGivenPoints)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat4("pnt", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_Impulse(float* param)
{
    if(ImGui::RadioButton("Impulse", *m_function == Function::Impulse))
    {
        *m_function = Function::Impulse;
        param[0] = 2;
    }

    if(*m_function == Function::Impulse)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("order", param, 0, 20);
    } 
}

void ShapingFunctions::_ShapingFunction_CubicPulse(float* param)
{
    if(ImGui::RadioButton("CubicPulse", *m_function == Function::CubicPulse))
    {
        *m_function = Function::CubicPulse;
        param[0] = 0.5;
        param[1] = 0.1;
    }

    if(*m_function == Function::CubicPulse)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_ExponentialStep(float* param)
{
    if(ImGui::RadioButton("ExponentialStep", *m_function == Function::ExponentialStep))
    {
        *m_function = Function::ExponentialStep;
        param[0] = 10;
        param[1] = 1;
    }

    if(*m_function == Function::ExponentialStep)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat("order", param, 0, 10);
        ImGui::SameLine();
        ImGui::SliderFloat("threshold", param+1, 0, 1);
    } 
}

void ShapingFunctions::_ShapingFunction_Parabola(float* param)
{
    if(ImGui::RadioButton("Parabola", *m_function == Function::Parabola))
    {
        *m_function = Function::Parabola;
        param[0] = 1;
    }

    if(*m_function == Function::Parabola)
    {
        ImGui::PushItemWidth(400);
        ImGui::SameLine(300);
        ImGui::SliderFloat("order", param, -20, 20);
    } 
}

void ShapingFunctions::_ShapingFunction_PowerCurve(float* param)
{
    if(ImGui::RadioButton("PowerCurve", *m_function == Function::PowerCurve))
    {
        *m_function = Function::PowerCurve;
        param[0] = 3;
        param[1] = 1;
    }

    if(*m_function == Function::PowerCurve)
    {
        ImGui::PushItemWidth(200);
        ImGui::SameLine(300);
        ImGui::SliderFloat2("pnt", param, 0, 10);
    } 
}
