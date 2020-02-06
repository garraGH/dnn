/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : layer.cpp
* author      : Garra
* time        : 2019-09-26 17:39:45
* description : 
*
============================================*/


#include "layer.h"
#include "imgui.h"

void Layer::OnImGuiRender()
{
    ImGui::Begin("Profile");
    for(auto pr : m_profileResults)
    {
        char label[50];
        strcpy(label, "%.3fms ");
        strcat(label, pr.Name.c_str());
        ImGui::Text(label, pr.Time);
    }
    m_profileResults.clear();

    ImGui::End();
}
