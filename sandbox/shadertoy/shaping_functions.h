/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/shadertoy/shaping_functions.h
* author      : Garra
* time        : 2019-10-13 09:51:31
* description : 
*
============================================*/


#pragma once
#include "shadertoy.h"

class ShapingFunctions : public ShaderToy
{
public:
    enum class Function
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

    ShapingFunctions();

    virtual std::string GetName() const override { return "ShapingFunctions"; }
    virtual void OnImGuiRender() override;
    virtual std::shared_ptr<Material> GetMaterial() const override;
    virtual std::shared_ptr<Shader> GetShader() const override;

    static std::shared_ptr<ShapingFunctions> Create();

private:
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
    Function* m_function;
};
