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
// #include "imgui_impl_glfw.h"
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
//     ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiLayer::OnAttach()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui::StyleColorsDark();
//     ImGui_ImplGlfw_InitForOpenGL(((X11Window*)Application::Get()->GetWindow())->GetInnerWindow(), true);
//
    ImGuiIO& io = ImGui::GetIO();
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;         // We can honor GetMouseCursor() values (optional)
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;          // We can honor io.WantSetMousePos requests (optional, rarely used)
    io.BackendPlatformName = "imgui_impl_elsa";

    // Temporary: should eventually use Elsa key codes.
    // Keyboard mapping. ImGui will use those indices to peek into the io.KeysDown[] array.
    io.KeyMap[ImGuiKey_Tab] = GLFW_KEY_TAB;
    io.KeyMap[ImGuiKey_LeftArrow] = GLFW_KEY_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = GLFW_KEY_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow] = GLFW_KEY_UP;
    io.KeyMap[ImGuiKey_DownArrow] = GLFW_KEY_DOWN;
    io.KeyMap[ImGuiKey_PageUp] = GLFW_KEY_PAGE_UP;
    io.KeyMap[ImGuiKey_PageDown] = GLFW_KEY_PAGE_DOWN;
    io.KeyMap[ImGuiKey_Home] = GLFW_KEY_HOME;
    io.KeyMap[ImGuiKey_End] = GLFW_KEY_END;
    io.KeyMap[ImGuiKey_Insert] = GLFW_KEY_INSERT;
    io.KeyMap[ImGuiKey_Delete] = GLFW_KEY_DELETE;
    io.KeyMap[ImGuiKey_Backspace] = GLFW_KEY_BACKSPACE;
    io.KeyMap[ImGuiKey_Space] = GLFW_KEY_SPACE;
    io.KeyMap[ImGuiKey_Enter] = GLFW_KEY_ENTER;
    io.KeyMap[ImGuiKey_Escape] = GLFW_KEY_ESCAPE;
    io.KeyMap[ImGuiKey_KeyPadEnter] = GLFW_KEY_KP_ENTER;
    io.KeyMap[ImGuiKey_A] = GLFW_KEY_A;
    io.KeyMap[ImGuiKey_C] = GLFW_KEY_C;
    io.KeyMap[ImGuiKey_V] = GLFW_KEY_V;
    io.KeyMap[ImGuiKey_X] = GLFW_KEY_X;
    io.KeyMap[ImGuiKey_Y] = GLFW_KEY_Y;
    io.KeyMap[ImGuiKey_Z] = GLFW_KEY_Z;

    ImGui_ImplOpenGL3_Init("#version 130");
}

void ImGuiLayer::OnDetach()
{

}

void ImGuiLayer::OnUpdate()
{
//     TRACE("ImGuiLayer::OnUpdate");
    ImGuiIO& io = ImGui::GetIO();
    float time = (float)glfwGetTime();
    io.DeltaTime = m_time>0.0f? (time-m_time) : 1.0f/60;
    m_time = time;

    Application* app = Application::Get();
    io.DisplaySize = ImVec2(app->GetWindow()->GetWidth(), app->GetWindow()->GetHeight());

    
    ImGui_ImplOpenGL3_NewFrame();
//     ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    static bool show = true;
    ImGui::ShowDemoWindow(&show);

    ImGui::Begin("IMGUI");
    ImGui::Button("Hello Dear ImGui");
    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiLayer::OnEvent(Event& e)
{
    INFO("ImGuiLayer::OnEvent {}", e);
    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(std::bind(&ImGuiLayer::_On##event, this, std::placeholders::_1))
    DISPATCH(MouseButtonPressedEvent);
    DISPATCH(MouseButtonReleasedEvent);
    DISPATCH(MouseMovedEvent);
    DISPATCH(MouseScrolledEvent);
    DISPATCH(KeyPressedEvent);
    DISPATCH(KeyReleasedEvent);
    DISPATCH(KeyTypedEvent);
    DISPATCH(WindowResizeEvent);
#undef DISPATCH
}

#define _ON(event) bool ImGuiLayer::_On##event(event& e)
_ON(MouseButtonPressedEvent)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDown[e.GetMouseButton()] = true;
    return false;
}

_ON(MouseButtonReleasedEvent)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDown[e.GetMouseButton()] = false;
    return false;
}

_ON(MouseMovedEvent)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(e.GetX(), e.GetY());
    return false;
}

_ON(MouseScrolledEvent)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH += e.GetOffsetX();
    io.MouseWheel  += e.GetOffsetY();
    return false;
}

_ON(KeyPressedEvent)
{
    ImGuiIO& io = ImGui::GetIO();
    io.KeysDown[e.GetKeyCode()] = true;

    io.KeyCtrl = io.KeysDown[GLFW_KEY_LEFT_CONTROL] || io.KeysDown[GLFW_KEY_RIGHT_CONTROL];
    io.KeyShift = io.KeysDown[GLFW_KEY_LEFT_SHIFT] || io.KeysDown[GLFW_KEY_RIGHT_SHIFT];
    io.KeyAlt = io.KeysDown[GLFW_KEY_LEFT_ALT] || io.KeysDown[GLFW_KEY_RIGHT_ALT];
    io.KeySuper = io.KeysDown[GLFW_KEY_LEFT_SUPER] || io.KeysDown[GLFW_KEY_RIGHT_SUPER];

    return false;
}

_ON(KeyReleasedEvent)
{
    ImGuiIO& io = ImGui::GetIO();
    io.KeysDown[e.GetKeyCode()] = false;
    return false;
}

_ON(KeyTypedEvent)
{
    ImGuiIO& io = ImGui::GetIO();
    int keyCode = e.GetKeyCode();
    if(keyCode>0 && keyCode<0x10000)
    {
        io.AddInputCharacter((unsigned short)keyCode);
    }
    return false;
}

_ON(WindowResizeEvent)
{
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(e.GetWidth(), e.GetHeight());
    io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
    glViewport(0, 0, e.GetWidth(), e.GetHeight());
    return false;
}

#undef _ON
