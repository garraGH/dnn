/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : layer_imgui.cpp
* author      : Garra
* time        : 2019-09-26 22:15:03
* description : 
*
============================================*/


#include "layer_imgui.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "../app/application.cpp"
#include "glfw3.h"

ImGuiLayer::ImGuiLayer()
    : Layer( "ImGuiLayer" )
    , m_time(0.0f)
{

}

ImGuiLayer::~ImGuiLayer()
{
}

void ImGuiLayer::OnAttach()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
//     ImGui_ImplGlfw_InitForOpenGL(Application::GetWindow(), true);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
}

void ImGuiLayer::OnDetach()
{

}

void ImGuiLayer::OnUpdate()
{
    ImGuiIO& io = ImGui::GetIO();
    float time = (float)glfwGetTime();
    io.DeltaTime = m_time>0.0f? (time-m_time) : 1.0f/60;
    m_time = time;

    Application& app = Application::Get();
    io.DisplaySize = ImVec2(app.GetWindow().GetWidth(), app.GetWindow().GetHeight());

    
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

//     static bool show = true;
//     ImGui::ShowDemoWindow(&show);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiLayer::OnEvent(Event& e)
{

}
