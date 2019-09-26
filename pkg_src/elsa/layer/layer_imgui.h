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

class ImGuiLayer : public Layer
{
public:
    ImGuiLayer();
    ~ImGuiLayer();

    void OnAttach() override;
    void OnDetach() override;

    void OnUpdate() override;
    void OnEvent(Event& e) override;

protected:
private:
    float m_time;
};
