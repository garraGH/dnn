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
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "../app/application.h"
#include "../window/window_x11.h"
#include "logger.h"
#include "../renderer/renderer.h"

std::shared_ptr<ImGuiLayer> ImGuiLayer::Create()
{
    return std::make_shared<ImGuiLayer>();
}

ImGuiLayer::ImGuiLayer()
    : Layer( "ImGuiLayer" )
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
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    if(io.ConfigFlags&ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    m_window = Application::GetInstance()->GetWindow();
    ImGui_ImplGlfw_InitForOpenGL(static_cast<GLFWwindow*>(m_window->GetNativeWindow()), true);
    ImGui_ImplOpenGL3_Init("#version 460");
}

void ImGuiLayer::OnDetach()
{

}

bool ImGuiLayer::CaptureInput() const
{
    const ImGuiIO& io = ImGui::GetIO();
    return io.WantTextInput || io.WantCaptureMouse || io.WantCaptureKeyboard;
}
void ImGuiLayer::Begin()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::End()
{
    ImGuiIO& io = ImGui::GetIO();
    int* size = m_window->GetSize();
    io.DisplaySize = ImVec2(size[0], size[1]);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if(io.ConfigFlags&ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}

void ImGuiLayer::OnImGuiRender()
{
//     static bool show = true;
//     ImGui::ShowDemoWindow(&show);


    static Renderer::PolygonMode s_polygonMode = Renderer::PolygonMode::FILL;

    ImGui::Begin("Application");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Separator();
    if(ImGui::RadioButton("POINT", s_polygonMode == Renderer::PolygonMode::POINT) && s_polygonMode != Renderer::PolygonMode::POINT)
    {
        s_polygonMode = Renderer::PolygonMode::POINT;
        Renderer::SetPolygonMode(s_polygonMode);
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("LINE", s_polygonMode == Renderer::PolygonMode::LINE) && s_polygonMode != Renderer::PolygonMode::LINE)
    {
        s_polygonMode = Renderer::PolygonMode::LINE;
        Renderer::SetPolygonMode(s_polygonMode);
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("FILL", s_polygonMode == Renderer::PolygonMode::FILL) && s_polygonMode != Renderer::PolygonMode::FILL)
    {
        s_polygonMode = Renderer::PolygonMode::FILL;
        Renderer::SetPolygonMode(s_polygonMode);
    }
    ImGui::Separator();
    if(ImGui::RadioButton("VerticalSync", m_window->IsVSync()))
    {
        m_window->SwitchVSync();
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("Fullscreen", m_window->IsFullscreen()))
    {
        m_window->SwitchFullscreen();
    }
    ImGui::Separator();
    ImGui::PushItemWidth(200);
    if(ImGui::InputInt2("Pos", m_window->GetPos(), ImGuiInputTextFlags_EnterReturnsTrue))
    {
        m_window->UpdatePos();
    }
    if(ImGui::InputInt2("Size", m_window->GetSize(), ImGuiInputTextFlags_EnterReturnsTrue))
    {
        m_window->UpdateSize();
    }
    ImGui::End();
}

