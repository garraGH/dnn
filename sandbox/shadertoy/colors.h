/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/shadertoy/colors.h
* author      : Garra
* time        : 2019-10-13 11:05:49
* description : 
*
============================================*/

#pragma once
#include "shadertoy.h"
#include "time.h"

class Colors : public ShaderToy
{
public:
    enum class Direction
    {
        Clockwise = 1, 
        CounterClockwise = -1, 
    };

    enum class Test
    {
        ColorTransition, 
        TextureTransition, 
        TextureMixEachChannel, 
        ColorFlagsAnimation,
        ColorFlagsShrink, 
    };

    enum class FlagLayout
    {
        Vertical, 
        Horizontal, 
        Circle, 
        Rainbow, 
    };

    enum class EasingFunction
    {
         Linear,
         ExponentialIn,
         ExponentialOut,
         ExponentialInOut,
         SineIn,
         SineOut,
         SineInOut,
         QinticIn,
         QinticOut,
         QinticInOut,
         QuarticIn,
         QuarticOut,
         QuarticInOut,
         QuadraticIn,
         QuadraticOut,
         QuadraticInOut,
         CubicIn,
         CubicOut,
         CubicInOut,
         ElasticIn,
         ElasticOut,
         ElasticInOut,
         CircularIn,
         CircularOut,
         CircularInOut,
         BounceIn,
         BounceOut,
         BounceInOut,
         BackIn,
         BackOut,
         BackInOut,
    };

public:
    Colors();

    virtual std::string GetName() const override { return "Colors"; }
    virtual std::shared_ptr<Material> GetMaterial() const override;
    virtual std::shared_ptr<Shader> GetShader() const override;
    virtual void OnUpdate(float deltaTime) override;
    virtual void OnImGuiRender() override;

    static std::shared_ptr<Colors> Create();

private:
    void _PrepareResources();
    void _CreateColorTransitionGUI();
    void _CreateTextureTransitionGUI();
    void _CreateTextureMixEachChannelGUI();
    void _CreateColorFlagsAnimationGUI();
    void _CreateColorFlagsShrink();
    void _CreateEasingFunctionsGUI();

private:
    std::unique_ptr<TimerCPU> m_timer = std::make_unique<TimerCPU>("ColorAnimation");
    float* m_speed = nullptr;
    EasingFunction* m_easingFunction = nullptr;
    Test* m_test = nullptr;
    float* m_lineWidth = nullptr;
    int* m_nFlags = nullptr;
    FlagLayout* m_flagLayout = nullptr;
    Direction* m_direction = nullptr;
};
