/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/shadertoy/hud.h
* author      : Garra
* time        : 2019-10-17 21:19:18
* description : 
*
============================================*/


#pragma once
#include "shadertoy.h"

class HUD : public ShaderToy
{
public:
    HUD();

    virtual std::string GetName() const { return "HUD"; }
    virtual void OnUpdate(float deltaTime) override;
    virtual void OnEvent(Event& e) override;
    virtual void OnImGuiRender() override;
    virtual std::shared_ptr<Material> GetMaterial() const override;
    virtual std::shared_ptr<Shader> GetShader() const override;

    static std::shared_ptr<HUD> Create();

protected:
#define ON(event) bool _On##event(event& e)
    ON(MouseButtonPressedEvent);
    ON(MouseMovedEvent);
    ON(MouseButtonReleasedEvent);
    ON(MouseScrolledEvent);
#undef ON

private:
    void _PrepareResources();

private:
    float m_speed = 1.0;
    float m_radRotated = 0;
    float* m_radarRange = nullptr;
    float* m_circleRadius = nullptr;
    float* m_circleCenter = nullptr;
    float* m_circleColor = nullptr;
    float* m_time = nullptr;

    std::unique_ptr<TimerCPU> m_timer = std::make_unique<TimerCPU>("HUD");
};

