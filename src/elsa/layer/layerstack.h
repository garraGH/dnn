/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/pkg_src/elsa/layer/layerstack.h
* author      : Garra
* time        : 2019-09-26 17:43:19
* description : 
*
============================================*/


#pragma once

#include "layer.h"
#include <vector>

class LayerStack
{
public:
    LayerStack();
    ~LayerStack();

    void PushLayer(Layer* layer);
    void PopLayer(Layer* layer);
    void PushOverlay(Layer* overlay);
    void PopOverlay(Layer* overlay);

    auto begin() { return m_layers.begin(); }
    auto end() { return m_layers.end(); }

private:
    std::vector<Layer*> m_layers;
    std::vector<Layer*>::iterator m_layerInsert;
};
