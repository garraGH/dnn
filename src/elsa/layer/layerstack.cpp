/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : layerstack.cpp
* author      : Garra
* time        : 2019-09-26 17:43:17
* description : 
*
============================================*/


#include "layerstack.h"

std::unique_ptr<LayerStack> LayerStack::Create()
{
    return std::make_unique<LayerStack>();
}

LayerStack::LayerStack()
{
}

LayerStack::~LayerStack()
{
}

void LayerStack::PushLayer(const std::shared_ptr<Layer>& layer)
{
    m_layers.emplace(m_layers.begin()+m_layerInsertIndex, layer);
    m_layerInsertIndex++;
}

void LayerStack::PushOverlay(const std::shared_ptr<Layer>& overlay)
{
    m_layers.emplace_back(overlay);
}

void LayerStack::PopLayer(const std::shared_ptr<Layer>& layer)
{
    auto it = std::find(begin(), end(), layer);
    if(it != end())
    {
        m_layers.erase(it);
        m_layerInsertIndex--;
    }
}

void LayerStack::PopOverlay(const std::shared_ptr<Layer>& overlay)
{
    auto it = std::find(begin(), end(), overlay);
    if(it != end())
    {
        m_layers.erase(it);
    }
}
