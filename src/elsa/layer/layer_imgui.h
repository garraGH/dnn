/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : layer_imgui.h
* author      : Garra
* time        : 2019-09-26 22:15:02
* description : 
*
============================================*/


#pragma once
#include "layer.h"
#include "../event/event_key.h"
#include "../event/event_mouse.h"
#include "../event/event_application.h"


class ImGuiLayer : public Layer
{
public:
    ImGuiLayer();
    ~ImGuiLayer();

    void OnAttach() override;
    void OnDetach() override;

    void OnUpdate() override;
    void OnEvent(Event& e) override;

private:
#define ON(event) bool _On##event(event& e)
    ON(MouseButtonPressedEvent);
    ON(MouseButtonReleasedEvent);
    ON(MouseMovedEvent);
    ON(MouseScrolledEvent);
    ON(KeyPressedEvent);
    ON(KeyReleasedEvent);
    ON(KeyTypedEvent);
    ON(WindowResizeEvent);
#undef ON



private:
    float m_time;
};
