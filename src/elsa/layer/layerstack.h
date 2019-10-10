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

    void PushLayer(const std::shared_ptr<Layer>& layer);
    void PopLayer(const std::shared_ptr<Layer>& layer);
    void PushOverlay(const std::shared_ptr<Layer>& overlay);
    void PopOverlay(const std::shared_ptr<Layer>& overlay);

    auto begin() { return m_layers.begin(); }
    auto end() { return m_layers.end(); }

    static std::unique_ptr<LayerStack> Create();

private:
    std::vector<std::shared_ptr<Layer>> m_layers;
    unsigned int m_layerInsertIndex = 0;
};
