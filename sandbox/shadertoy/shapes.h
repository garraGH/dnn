/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/shadertoy/shapes.h
* author      : Garra
* time        : 2019-10-13 11:05:54
* description : 
*
============================================*/


#pragma once
#include "shadertoy.h"

class Shapes : public ShaderToy
{
public:
    enum class Style
    {
        Line = 0, 
        Rectangle, 
        Circle, 
        Elipse, 
        Ring, 
    };

    enum class Mode
    {
        Fill = 0, 
        Outline, 
        Both, 
    };

    struct Colors
    {
        glm::vec4 cFilled;
        glm::vec4 cOutline;
    };

    struct Points
    {
        glm::vec2 pnt1;
        glm::vec2 pnt2;
        glm::vec2 pnt3;
    };



public:
    Shapes();

    virtual std::string GetName() const { return "Shapes"; }
    virtual void OnImGuiRender() override;
    virtual std::shared_ptr<Material> GetMaterial() const override;
    virtual std::shared_ptr<Shader> GetShader() const override;

    virtual void OnEvent(Event& e) override;

    static std::shared_ptr<Shapes> Create();

protected:
#define ON(event) bool _On##event(event& e)
    ON(MouseButtonPressedEvent);
    ON(MouseMovedEvent);
    ON(MouseButtonReleasedEvent);
#undef ON

private:
    void _PrepareResources();

    Style* m_style = nullptr;
    Mode* m_mode = nullptr;
    Colors* m_colors = nullptr;
    Points* m_points = nullptr;
    float* m_lineWidth = nullptr;

    bool m_bPressed = false;
};
