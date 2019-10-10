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

    void Begin();
    void OnImGuiRender() override;
    void End();

    static std::shared_ptr<ImGuiLayer> Create();
};
