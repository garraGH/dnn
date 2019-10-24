/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/src/elsa/renderer/viewport/viewport.h
* author      : Garra
* time        : 2019-10-25 11:47:33
* description : 
*
============================================*/


#pragma once
#include <array>
#include <memory>
#include "../camera/camera.h"
#include "../../event/event_application.h"

class Viewport 
{
public:
    enum class Type
    {
        Percentage, 
        Fixed, 
    };


public:
    Viewport(const std::string& name);
    void SetType(Type t);
    void SetRange(float left, float bottom, float width, float height);
    std::array<float, 4> GetRange() const;

    void OnUpdate(float deltaTime);
    void OnEvent(Event& e);
    void OnImGuiRender();
    const std::shared_ptr<Camera>& GetCamera() const;

    static std::shared_ptr<Viewport> Create(const std::string& name); 

protected:
#define ON(event) bool _On##event(event& e)
    ON(WindowResizeEvent);
#undef ON
private:
    std::string m_name;
    Type m_type = Type::Percentage;
    std::array<float, 2> m_windowSize = {1000, 1000};
    std::array<float, 4> m_range = {0, 0, 1, 1};
    std::shared_ptr<Camera> m_camera = nullptr;
};
