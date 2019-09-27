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
#include "../window/window_x11.h"

ImGuiLayer::ImGuiLayer()
    : Layer( "ImGuiLayer" )
    , m_time(0.0f)
{

}

ImGuiLayer::~ImGuiLayer()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiLayer::OnAttach()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(((X11Window*)Application::Get()->GetWindow())->GetInnerWindow(), true);
    ImGui_ImplOpenGL3_Init("#version 130");
}

void ImGuiLayer::OnDetach()
{

}

void ImGuiLayer::OnUpdate()
{
    TRACE("ImGuiLayer::OnUpdate");
    ImGuiIO& io = ImGui::GetIO();
    float time = (float)glfwGetTime();
    io.DeltaTime = m_time>0.0f? (time-m_time) : 1.0f/60;
    m_time = time;

    Application* app = Application::Get();
    io.DisplaySize = ImVec2(app->GetWindow()->GetWidth(), app->GetWindow()->GetHeight());

    
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    static bool show = true;
    ImGui::ShowDemoWindow(&show);

    ImGui::Begin("IMGUI");
    ImGui::Button("Hello");
    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiLayer::OnEvent(Event& e)
{

}
