/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/pkg_src/elsa/layer/layer.h
* author      : Garra
* time        : 2019-09-26 17:39:51
* description : 
*
============================================*/


#pragma once

#include <vector>
#include "core.h"
#include "../event/event.h"


#define PROFILE_TO_IMGUI(name) TimerCPU timer##__LINE__(name, false, [&](const std::string& taskName, float t) { m_profileResults.push_back({taskName, t*1000}); })

class Layer
{
public:
    Layer(const std::string& name = "Layer") : m_name(name) {}
    virtual ~Layer() {}

    virtual void OnAttach() {}
    virtual void OnDetach() {}
    virtual void OnUpdate(float deltaTime) {}
    virtual void OnEvent(Event& e) {}
    virtual void OnImGuiRender();

    inline const std::string& GetName() const { return m_name; }
    

protected:
    std::string m_name;
    struct ProfileResult
    {
        std::string Name;
        float Time;
    };

    std::vector<ProfileResult> m_profileResults;
};
