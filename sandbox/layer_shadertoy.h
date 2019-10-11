/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/layer_shadertoy.h
* author      : Garra
* time        : 2019-10-10 14:26:51
* description : 
*
============================================*/


#pragma once
#include "elsa.h"

class ShaderToyLayer : public Layer
{
public:
    enum class TestShader
    {
        FlatColor, 
        ShapingFunctions, 
        Shapes, 
    };
    enum class ShapingFunction
    {   
        Linear = 0, 
        Step, 
        SmoothStep, 
        Power, 
        Sine,
        Cosine, 
        BlinnWyvillCosineApproximation, 
        DoubleCubicSeat, 
        DoubleCubicSeatWidthLinearBlend, 
        DoubleOddPolynomialSeat, 
        SymmetricDoublePolynomialSigmoids, 
        QuadraticThroughGivenPoint, 
        ExponentialEaseIn, 
        ExponentialEaseOut, 
        ExponentialEasing, 
        DoubleExponentialSeat, 
        DoubleExponentialSigmoid, 
        LogisticSigmoid, 
        CircularEaseIn, 
        CircularEaseOut, 
        DoubleCircelSeat, 
        DoubleCircleSigmoid, 
        DoubleEllipticSeat, 
        DoubleEllipticSigmoid, 
        DoubleLinearWidthCircularFillet, 
        CircularArcThroughGivenPoint, 
        QuadraticBezier, 
        CubicBezier, 
        CubicBezierThroughTwoGivenPoints, 
        Impulse, 
        CubicPulse, 
        ExponentialStep, 
        Parabola, 
        PowerCurve,  
    };
public:
    ShaderToyLayer();

    virtual void OnEvent(Event& e) override;
    virtual void OnUpdate(float deltaTime) override;
    virtual void OnImGuiRender() override;

    static std::shared_ptr<ShaderToyLayer> Create();

protected:
    void _PrepareResources();

private:
    std::unique_ptr<CameraContoller> m_cameraController = std::make_unique<CameraContoller>(Camera::Type::Orthographic);
    std::shared_ptr<Shader> m_currentShader = nullptr;
    std::shared_ptr<Renderer::Element> m_canvas = nullptr;

    TestShader m_testShader = TestShader::Shapes;
    ShapingFunction m_shapingFunction = ShapingFunction::Linear;
};
