/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/shadertoy/matrix.h
* author      : Garra
* time        : 2019-10-17 16:11:06
* description : 
*
============================================*/

#pragma once
#include "shadertoy.h"

class Matrix : public ShaderToy
{
public:
    enum class Style
    {
        T = 0,     // Translate
        R,         // Rotate
        S,         // Scale
        TR, 
        TS, 
        RS, 
        TRS, 
    };


public:
    Matrix();

    virtual std::string GetName() const { return "Matrix"; }
    virtual void OnUpdate() override;
    virtual void OnImGuiRender() override;
    virtual std::shared_ptr<Material> GetMaterial() const override;
    virtual std::shared_ptr<Shader> GetShader() const override;

    virtual void OnEvent(Event& e) override;

    static std::shared_ptr<Matrix> Create();

protected:
#define ON(event) bool _On##event(event& e)
    ON(MouseButtonPressedEvent);
    ON(MouseMovedEvent);
    ON(MouseButtonReleasedEvent);
    ON(MouseScrolledEvent);
#undef ON

private:
    void _PrepareResources();

    std::unique_ptr<TimerCPU> m_timer = std::make_unique<TimerCPU>("MatrixTransform");
    float* m_radius = nullptr;
    Style* m_style = nullptr;
    float* m_speed = nullptr;
    int m_showSDF = 0;
};
