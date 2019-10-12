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
        DoubleCircleSeat, 
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
    void _ShapingFunction_Linear(float* param);
    void _ShapingFunction_Step(float* param);
    void _ShapingFunction_SmoothStep(float* param);
    void _ShapingFunction_Power(float* param);
    void _ShapingFunction_Sine(float* param);
    void _ShapingFunction_Cosine(float* param);
    void _ShapingFunction_BlinnWyvillCosineApproximation(float* param);
    void _ShapingFunction_DoubleCubicSeat(float* param);
    void _ShapingFunction_DoubleCubicSeatWidthLinearBlend(float* param);
    void _ShapingFunction_DoubleOddPolynomialSeat(float* param, int* order);
    void _ShapingFunction_SymmetricDoublePolynomialSigmoids(float* param, int* order);
    void _ShapingFunction_QuadraticThroughGivenPoint(float* param);
    void _ShapingFunction_ExponentialEaseIn(float* param);
    void _ShapingFunction_ExponentialEaseOut(float* param);
    void _ShapingFunction_ExponentialEasing(float* param);
    void _ShapingFunction_DoubleExponentialSeat(float* param);
    void _ShapingFunction_DoubleExponentialSigmoid(float* param);
    void _ShapingFunction_LogisticSigmoid(float* param);
    void _ShapingFunction_CircularEaseIn(float* param);
    void _ShapingFunction_CircularEaseOut(float* param);
    void _ShapingFunction_DoubleCircleSeat(float* param);
    void _ShapingFunction_DoubleCircleSigmoid(float* param);
    void _ShapingFunction_DoubleEllipticSeat(float* param);
    void _ShapingFunction_DoubleEllipticSigmoid(float* param);
    void _ShapingFunction_DoubleLinearWidthCircularFillet(float* param);
    void _ShapingFunction_CircularArcThroughGivenPoint(float* param);
    void _ShapingFunction_QuadraticBezier(float* param);
    void _ShapingFunction_CubicBezier(float* param);
    void _ShapingFunction_CubicBezierThroughTwoGivenPoints(float* param);
    void _ShapingFunction_Impulse(float* param);
    void _ShapingFunction_CubicPulse(float* param);
    void _ShapingFunction_ExponentialStep(float* param);
    void _ShapingFunction_Parabola(float* param);
    void _ShapingFunction_PowerCurve(float* param);

private:
    std::unique_ptr<CameraContoller> m_cameraController = std::make_unique<CameraContoller>(Camera::Type::Orthographic);
    std::shared_ptr<Shader> m_currentShader = nullptr;
    std::shared_ptr<Renderer::Element> m_canvas = nullptr;

    TestShader m_testShader = TestShader::Shapes;
    ShapingFunction* m_shapingFunction = nullptr;
};
