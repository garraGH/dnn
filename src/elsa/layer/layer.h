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

#include "../event/event.h"

class Layer
{
public:
    Layer(const std::string& name = "Layer") : m_name(name) {}
    virtual ~Layer() {}

    virtual void OnAttach() {}
    virtual void OnDetach() {}
    virtual void OnUpdate() {}
    virtual void OnEvent(Event& e) {}
    virtual void OnImGuiRender() {}

    inline const std::string& GetName() const { return m_name; }

protected:
    std::string m_name;
};
