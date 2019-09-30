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

LayerStack::LayerStack()
{
}

LayerStack::~LayerStack()
{
    for(Layer* layer : m_layers)
    {
        delete layer;
    }
}

void LayerStack::PushLayer(Layer* layer)
{
    m_layers.emplace(m_layers.begin()+m_layerInsertIndex, layer);
    m_layerInsertIndex++;
}

void LayerStack::PushOverlay(Layer* overlay)
{
    m_layers.emplace_back(overlay);
}

void LayerStack::PopLayer(Layer* layer)
{
    auto it = std::find(begin(), end(), layer);
    if(it != end())
    {
        m_layers.erase(it);
        m_layerInsertIndex--;
    }
}

void LayerStack::PopOverlay(Layer* overlay)
{
    auto it = std::find(begin(), end(), overlay);
    if(it != end())
    {
        m_layers.erase(it);
    }
}
