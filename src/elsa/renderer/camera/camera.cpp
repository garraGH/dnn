/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : camera.cpp
* author      : Garra
* time        : 2019-10-03 20:18:41
* description : 
*
============================================*/


#include "camera.h"
#include "glm/gtc/matrix_transform.hpp"
#include "imgui.h"
#include "glm/gtx/string_cast.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "../../core.h"
#include "../../input/input.h"
#include "../../input/codes_mouse.h"
#include "../renderer.h"

std::shared_ptr<Camera> Camera::Create(const std::string& name, Type type, Usage usage)
{
    return std::make_shared<Camera>(name, type, usage);
}

void Camera::Sight::_Update()
{
    if(!m_dirty)
    {
        return;
    }
    m_dirty = false;

    m_type == Type::Orthographic? _calcOrthoMatrix() : _calcPerspectiveMatrix();
    m_matClip2View = glm::inverse(m_matView2Clip);
}

void Camera::Sight::_calcOrthoMatrix()
{
    m_matView2Clip[0][0] = 2.0/m_width/m_scale;
    m_matView2Clip[1][1] = m_asp*m_matView2Clip[0][0];
    m_matView2Clip[2][2] = -2.0/(m_far-m_near);
    m_matView2Clip[2][3] = 0;
    m_matView2Clip[3][2] = -(m_far+m_near)/(m_far-m_near);
    m_matView2Clip[3][3] = 1;
}

void Camera::Sight::_calcPerspectiveMatrix()
{
    m_matView2Clip[1][1] = 1.0/glm::tan(glm::radians(m_vfov)*0.5)/m_scale;
    m_matView2Clip[0][0] = m_matView2Clip[1][1]/m_asp;
    m_matView2Clip[2][2] = -(m_far+m_near)/(m_far-m_near);
    m_matView2Clip[2][3] = -1;
    m_matView2Clip[3][2] = -2.0*m_far*m_near/(m_far-m_near);
    m_matView2Clip[3][3] = 0;
}

void Camera::Sight::OnImGuiRender()
{
    ImGui::PushItemWidth(200);
    ImGui::Separator();
    if(ImGui::RadioButton("Orthographic", m_type == Type::Orthographic) && m_type != Type::Orthographic)
    {
        m_type = Type::Orthographic;
        m_dirty = true;
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("Perspective", m_type == Type::Perspective) && m_type != Type::Perspective)
    {
        m_type = Type::Perspective;
        m_dirty = true;
    }

    if(m_type == Type::Orthographic)
    {
        m_dirty |= ImGui::SliderFloat("width", &m_width, 0, 1000);
    }
    else
    {
        m_dirty |= ImGui::SliderFloat("vFov", &m_vfov, 0, 180);
    }
    ImGui::SameLine();
    m_dirty |= ImGui::DragFloat("scale", &m_scale,  0.1,  0.1,  10);
    ImGui::SameLine();
    ImGui::Text("asp: %.3f", m_asp);
    m_dirty |= ImGui::DragFloat("near", &m_near,  0.1,  0.1, 10);
    ImGui::SameLine();
    m_dirty |= ImGui::DragFloat("far", &m_far,  10,  100,  10000);
}

void Camera::_UpdateWorld2View()
{
    if(!m_dirtyW2V)
    {
        return;
    }
    m_dirtyW2V = false;
//     glm::mat4 transform = glm::rotate(glm::mat4(1.0f), glm::radians(m_orientation.z), glm::vec3(0, 0, 1)) 
//                         * glm::rotate(glm::mat4(1.0f), glm::radians(m_orientation.x), glm::vec3(1, 0, 0))   
//                         * glm::rotate(glm::mat4(1.0f), glm::radians(m_orientation.y), glm::vec3(0, 1, 0)) 
//                         * glm::translate(glm::mat4(1.0f), m_position);
//     if(m_usage == Usage::TwoDimension)
//     {
//         transform = glm::rotate(glm::mat4(1.0f), glm::radians(m_orientation.z), glm::vec3(0, 0, 1)) * glm::translate(glm::mat4(1.0f), m_position);
//     }
// 
//     float x = -glm::radians(m_orientation.x);
//     float y = -glm::radians(m_orientation.y);
//     float z = -glm::radians(m_orientation.z);
//     float cosx = cos(x);
//     float sinx = sin(x);
//     float cosy = cos(y);
//     float siny = sin(y);
//     float cosz = cos(z);
//     float sinz = sin(z);
//     glm::mat4 yaw = 
//     {
//         +cosy,     0, -siny,     0, 
//             0,     1,     0,     0, 
//         +siny,     0, +cosy,     0, 
//             0,     0,     0,     1
//     };
//     glm::mat4 pitch = 
//     {
//         1,     0,     0,     0, 
//         0, +cosx, +sinx,     0, 
//         0, -sinx, +cosx,     0, 
//         0,     0,     0,     1
//     };
//     glm::mat4 roll = 
//     {
//         +cosz, +sinz,     0,     0, 
//         -sinz, +cosz,     0,     0, 
//             0,     0,     1,     0, 
//             0,     0,     0,     1
//     };
    glm::mat4 trans = 
    {
        1, 0, 0, 0, 
        0, 1, 0, 0, 
        0, 0, 1, 0,
        -m_position.x, -m_position.y, -m_position.z, 1
    };

//     m_matView = roll*pitch*yaw*trans;

    //lookat
    glm::vec3 D = -GetDirection();
    glm::vec3 R = glm::normalize(glm::cross(m_up, D));
    glm::vec3 U = glm::cross(D, R);
    glm::mat4 rot = 
    {
        R.x, U.x, D.x, 0, 
        R.y, U.y, D.y, 0, 
        R.z, U.z, D.z, 0, 
        0, 0, 0, 1
    };
    m_matWorld2View = rot*trans;



}
    
void Camera::_UpdateView2World()
{
//     if(!m_dirtyV2W)
//     {
//         return;
//     }
//     m_dirtyV2W = false;

    m_matView2World = glm::inverse(World2View());
}

void Camera::_UpdateWorld2Clip()
{
//     if(!m_dirtyW2C)
//     {
//         return;
//     }
//     m_dirtyW2C = false;

    _UpdateWorld2View();
    m_matWorld2Clip = m_sight.View2Clip()*m_matWorld2View;
}

void Camera::_UpdateClip2World()
{
//     if(!m_dirtyC2W)
//     {
//         return;
//     }
//     m_dirtyC2W = false;

    m_matClip2World = glm::inverse(World2Clip());
}

std::array<glm::vec3, 4> Camera::_GetCornersInWorldSpace(float d)
{
    float y = d*std::tan(glm::radians(m_sight.GetVfov()*0.5));
    float x = m_sight.GetAsp()*y;
//     glm::vec3 lb = View2World()*glm::vec4(-x, -y, -d, 1);
//     glm::vec3 rb = View2World()*glm::vec4(+x, -y, -d, 1);
//     glm::vec3 rt = View2World()*glm::vec4(+x, +y, -d, 1);
//     glm::vec3 lt = View2World()*glm::vec4(-x, +y, -d, 1);
//
    glm::vec3 lb = View2World({-x, -y, -d});
    glm::vec3 rb = View2World({+x, -y, -d});
    glm::vec3 rt = View2World({+x, +y, -d});
    glm::vec3 lt = View2World({-x, +y, -d});
    return { lb, rb, rt, lt };
}

std::array<glm::vec3, 4> Camera::GetNearCornersInWorldSpace()
{
    return _GetCornersInWorldSpace(m_sight.GetNear());
}

std::array<glm::vec3, 4> Camera::GetFarCornersInWorldSpace()
{
    return _GetCornersInWorldSpace(m_sight.GetFar());
}

void Camera::OnUpdate(float deltaTime)
{
    if(Input::IsKeyPressed(KEY_x))
    {
        Translate({-deltaTime*m_speedTrans, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_X))
    {
        Translate({+deltaTime*m_speedTrans, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_y))
    {
        Translate({0, -deltaTime*m_speedTrans, 0});
    }
    if(Input::IsKeyPressed(KEY_Y))
    {
        Translate({0, +deltaTime*m_speedTrans, 0});
    }
    if(Input::IsKeyPressed(KEY_z))
    {
        INFO("Camera::OnUpdate: KEY_z Pressed.");
        Translate({0, 0, -deltaTime*m_speedTrans});
    }
    if(Input::IsKeyPressed(KEY_Z))
    {
        INFO("Camera::OnUpdate: KEY_Z Pressed.");
        Translate({0, 0, +deltaTime*m_speedTrans});
    }
    if(Input::IsKeyPressed(KEY_s))
    {
        Scale(-deltaTime*m_speedScale);
    }
    if(Input::IsKeyPressed(KEY_S))
    {
        Scale(+deltaTime*m_speedScale);
    }

    if(Input::IsKeyPressed(KEY_b))
    {
        Revert();
    }

    if(m_rotationEnabled)
    {
        if(Input::IsKeyPressed(KEY_w))
        {
            Rotate({-deltaTime*m_speedRotat, 0, 0});
        }
        if(Input::IsKeyPressed(KEY_W))
        {
            Rotate({+deltaTime*m_speedRotat, 0, 0});
        }
        if(Input::IsKeyPressed(KEY_e))
        {
            Rotate({0, -deltaTime*m_speedRotat, 0});
        }
        if(Input::IsKeyPressed(KEY_E))
        {
            Rotate({0, +deltaTime*m_speedRotat, 0});
        }
        if(Input::IsKeyPressed(KEY_r))
        {
            Rotate({0, 0, -deltaTime*m_speedRotat});
        }
        if(Input::IsKeyPressed(KEY_R))
        {
            Rotate({0, 0, +deltaTime*m_speedRotat});
        }
    }

    if(m_bMiddleButtonPressed)
    {
        Translate(m_speedOfRoamTranslation*deltaTime);
        SetTarget(_RotateAroundTargetOnHorizontalPlane(m_target, m_position, m_speedOfRoamRotation*deltaTime));
    }
}


void Camera::OnEvent(Event& e)
{
    EventDispatcher dispatcher(e);
    dispatcher.Dispatch<MouseScrolledEvent>(BIND_EVENT_CALLBACK(Camera, _OnMouseScrolled));
    dispatcher.Dispatch<MouseButtonPressedEvent>(BIND_EVENT_CALLBACK(Camera, _OnMouseButtonPressed));
    dispatcher.Dispatch<MouseButtonReleasedEvent>(BIND_EVENT_CALLBACK(Camera, _OnMouseButtonReleased));
    dispatcher.Dispatch<MouseMovedEvent>(BIND_EVENT_CALLBACK(Camera, _OnMouseMoved));
    dispatcher.Dispatch<WindowResizeEvent>(BIND_EVENT_CALLBACK(Camera, _OnWindowResize));
}

bool Camera::_OnMouseScrolled(MouseScrolledEvent& e)
{
    if(Input::IsKeyPressed(KEY_LEFT_CONTROL))
    {
        Scale(-e.GetOffsetY()*m_speedScale);
    }
    else
    {
        auto [x, y] = Input::GetMousePosition();
        glm::vec3 worldPosOfCurrentCursor = Screen2World({x, y});
        glm::vec3 offset = worldPosOfCurrentCursor-m_position;
        float distance = e.GetOffsetY()*glm::length(offset);
        glm::vec3 dir = glm::normalize(offset);
        Translate(dir*distance*m_speedScale);
    }
    return false;
}

bool Camera::_OnMouseButtonPressed(MouseButtonPressedEvent& e)
{
    m_bLeftButtonPressed = e.GetMouseButton() == MOUSE_BUTTON_LEFT;
    m_bMiddleButtonPressed = (e.GetMouseButton()==MOUSE_BUTTON_MIDDLE);
    m_bRightButtonPressed = e.GetMouseButton() == MOUSE_BUTTON_RIGHT;
    
    m_cursorScreenPntWhenButtonPressed = Input::GetMousePosition();
    m_worldPosOfCursorWhenButtonPressed = Screen2World({m_cursorScreenPntWhenButtonPressed.first, m_cursorScreenPntWhenButtonPressed.second});

    m_cameraWorldPositionWhenButtonPressed = m_position;
    m_cameraWorldTargetWhenButtonPressed = m_target;
    
    if(m_bMiddleButtonPressed)
    {
        _Roam();
    }
    return false;
}


#define PI 3.14159265
bool Camera::_OnMouseMoved(MouseMovedEvent& e)
{
    if(!(m_bLeftButtonPressed || m_bMiddleButtonPressed || m_bRightButtonPressed))
    {
        return false;
    }


    if(m_bRightButtonPressed)
    {
        _RotateAroundPosOfButtonPressed(e);
    }

    if(m_bMiddleButtonPressed)
    {
        _Roam();
    }

    if(m_bLeftButtonPressed)
    {
        _DragScene();
    }

    return false;
}

void Camera::_DragScene()
{
    glm::vec3 worldPosOfCurrentCursor = _GetWorldPosOfCurrentCursorOnDragPlane();
    glm::vec3 offset = worldPosOfCurrentCursor-m_worldPosOfCursorWhenButtonPressed;
    SetPosition(m_cameraWorldPositionWhenButtonPressed-offset);
    SetTarget(m_cameraWorldTargetWhenButtonPressed-offset);
}

void Camera::_Roam()
{
    glm::vec3 pos = m_position;
    glm::vec3 tar = m_target;

    SetPosition(m_cameraWorldPositionWhenButtonPressed);
    SetTarget(m_cameraWorldTargetWhenButtonPressed);


    glm::vec3 worldCenterPos = _GetWorldPosOnHorizontalPlane({m_windowSize[0]/2, m_windowSize[1]/2});
    if(Input::IsKeyPressed(KEY_LEFT_CONTROL))
    {
        unsigned int x = m_cursorScreenPntWhenButtonPressed.first;
        unsigned int y = Input::GetMouseY();
        m_speedOfRoamTranslation = _GetWorldPosOnHorizontalPlane({x, y})-worldCenterPos;
        m_speedOfRoamRotation = 2.0*Input::GetMouseX()/m_windowSize[0]-1.0;
    }
    else
    {
        glm::vec3 worldPosOfCurrentCursor = _GetWorldPosOfCurrentCursorOnHorizontalPlane();
        m_speedOfRoamTranslation = worldPosOfCurrentCursor-worldCenterPos;
        m_speedOfRoamRotation = 0;
    }

    SetPosition(pos);
    SetTarget(tar);
}

glm::vec3 Camera::_RotateAroundTargetOnHorizontalPlane(const glm::vec3& pos, const glm::vec3& target, float theta)
{
    float dx = target.x-pos.x;
    float dz = target.z-pos.z;
    float r = std::sqrt(dx*dx+dz*dz);
    float a = std::atan2(dz, dx)+theta;
    glm::vec3 posRotated = pos;
    posRotated.x = target.x-r*std::cos(a);
    posRotated.z = target.z-r*std::sin(a);
    return posRotated;
}

glm::vec3 Camera::_RotateAroundTargetOnVerticalPlane(const glm::vec3& pos, const glm::vec3& target, float theta)
{
    glm::vec3 posInView = World2View(pos);
    glm::vec3 targetInView = World2View(target);
    float dy = targetInView.y-posInView.y;
    float dz = targetInView.z-posInView.z;
    float r = std::sqrt(dy*dy+dz*dz);
    float a = std::atan2(dz, dy)+theta;
    glm::vec3 posRotatedInView = posInView;
    posRotatedInView.y = targetInView.y-r*std::cos(a);
    posRotatedInView.z = targetInView.z-r*std::sin(a);
    return View2World(posRotatedInView);
}                               


void Camera::_RotateAroundPosOfButtonPressed(MouseMovedEvent& e)
{
    SetPosition(m_cameraWorldPositionWhenButtonPressed);
    SetTarget(m_cameraWorldTargetWhenButtonPressed);

    float dx = +e.GetX()-m_cursorScreenPntWhenButtonPressed.first;
    float dy = -e.GetY()+m_cursorScreenPntWhenButtonPressed.second;
    dx *= PI/m_windowSize[0];
    dy *= PI/m_windowSize[1];
    if(Input::IsKeyPressed(KEY_LEFT_CONTROL)) // Rotate only, keep position(first person perspective)
    {
        // horizontal
        SetTarget(_RotateAroundTargetOnHorizontalPlane(m_target, m_position, dx));
        // vertical
        SetTarget(_RotateAroundTargetOnVerticalPlane(m_target, m_position, dy));
    }
    else                                      // Rotate around m_worldPosOfCursorWhenButtonPressed
    {
        // horizontal
        SetPosition(_RotateAroundTargetOnHorizontalPlane(m_position, m_worldPosOfCursorWhenButtonPressed, dx));
        SetTarget(_RotateAroundTargetOnHorizontalPlane(m_target, m_worldPosOfCursorWhenButtonPressed, dx));
        // vertical
        SetPosition(_RotateAroundTargetOnVerticalPlane(m_position, m_worldPosOfCursorWhenButtonPressed, dy));
        SetTarget(_RotateAroundTargetOnVerticalPlane(m_target, m_worldPosOfCursorWhenButtonPressed, dy));
    }

}

glm::vec3 Camera::_GetWorldPosOfCurrentCursorOnDragPlane()
{
    SetPosition(m_cameraWorldPositionWhenButtonPressed);
    SetTarget(m_cameraWorldTargetWhenButtonPressed);

    return Input::IsKeyPressed(KEY_LEFT_CONTROL)? _GetWorldPosOfCurrentCursorOnVerticalPlane() : _GetWorldPosOfCurrentCursorOnHorizontalPlane();
}

glm::vec3 Camera::_GetWorldPosOnHorizontalPlane(const glm::vec2& pntOnScreen)
{
    glm::vec3 worldPosOfCurrentCursor = Screen2World(pntOnScreen);
    glm::vec3 dirInWorldSpace = glm::normalize(worldPosOfCurrentCursor-m_cameraWorldPositionWhenButtonPressed);
    float f = (m_worldPosOfCursorWhenButtonPressed.y-m_cameraWorldPositionWhenButtonPressed.y)/dirInWorldSpace.y;

    glm::vec3 worldPosOnDragPlane;
    worldPosOnDragPlane.y = m_worldPosOfCursorWhenButtonPressed.y;
    worldPosOnDragPlane.x = dirInWorldSpace.x*f+m_cameraWorldPositionWhenButtonPressed.x;
    worldPosOnDragPlane.z = dirInWorldSpace.z*f+m_cameraWorldPositionWhenButtonPressed.z;

    return worldPosOnDragPlane;
}

glm::vec3 Camera::_GetWorldPosOfCurrentCursorOnHorizontalPlane()
{
    auto [x, y] = Input::GetMousePosition();
    return _GetWorldPosOnHorizontalPlane({x, y});
}

glm::vec3 Camera::_GetWorldPosOfCurrentCursorOnVerticalPlane()
{
    auto [x, y] = Input::GetMousePosition();
    glm::vec3 viewPosOfCursorPressed = World2View(m_worldPosOfCursorWhenButtonPressed);
    glm::vec3 viewPosOfCursorCurrent = Screen2View({x, y});
    glm::vec3 dirInViewSpace = glm::normalize(viewPosOfCursorCurrent);
    float f = viewPosOfCursorPressed.z/dirInViewSpace.z;

    glm::vec3 viewPosOnDragPlane;
    viewPosOnDragPlane.z = viewPosOfCursorPressed.z;
    viewPosOnDragPlane.x = dirInViewSpace.x*f;
    viewPosOnDragPlane.y = dirInViewSpace.y*f;

    return View2World(viewPosOnDragPlane);
}


bool Camera::_OnMouseButtonReleased(MouseButtonReleasedEvent& e)
{
    m_bLeftButtonPressed &= !(e.GetMouseButton() == MOUSE_BUTTON_LEFT);
    m_bMiddleButtonPressed &= !(e.GetMouseButton() == MOUSE_BUTTON_MIDDLE);
    m_bRightButtonPressed &= !(e.GetMouseButton() == MOUSE_BUTTON_RIGHT);

    if(!m_bMiddleButtonPressed)
    {
        m_speedOfRoamRotation = 0;
        m_speedOfRoamTranslation = glm::vec3(0.0f);
    }

    return false;
}

bool Camera::_OnWindowResize(WindowResizeEvent& e)
{
    m_windowSize = {e.GetWidth(), e.GetHeight()};
    return false;
}

float Camera::Sight::DepthClip2NDC(float depthInClip) const
{
    float A = -(m_far+m_near)/(m_far-m_near);
    float B = -2*m_far*m_near/(m_far-m_near);
    float depthInNDC = -(A*depthInClip+B)/depthInClip;
    return depthInNDC;
}

float Camera::Sight::DepthNDC2Clip(float depthInNDC) const
{
    float A = -(m_far+m_near)/(m_far-m_near);
    float B = -2*m_far*m_near/(m_far-m_near);
    float depthInClip = -B/(A+depthInNDC);
    return depthInClip;
}

glm::vec3 Camera::Screen2View(const glm::vec2& pntOnScreen)
{
    unsigned int w = m_windowSize[0];
    unsigned int h = m_windowSize[1];

    float x = pntOnScreen.x;
    float y = h-pntOnScreen.y;

    float z_ndc = Renderer::GetPixelDepth(x, y, m_frameBuffer)*2-1;
    float z_cs = m_sight.DepthNDC2Clip(z_ndc);
    glm::vec4 pos_ndc(x/w*2-1.0, y/h*2-1.0, z_ndc, 1.0);
    glm::vec4 pos_cs = -z_cs*pos_ndc;
    glm::vec4 pos_vs = Clip2View()*pos_cs;

    return pos_vs;
}

glm::vec2 Camera::View2Screen(const glm::vec3& posInView)
{
    glm::vec4 pos_cs = View2Clip()*glm::vec4(posInView, 1.0);
    pos_cs /= pos_cs.w;
    return { (pos_cs.x+1)*m_windowSize[0]/2, m_windowSize[1]-(pos_cs.y+1)*m_windowSize[1]/2 };
}

glm::vec3 Camera::Screen2World(const glm::vec2& pntOnScreen)
{
    unsigned int w = m_windowSize[0];
    unsigned int h = m_windowSize[1];

    float x = pntOnScreen.x;
    float y = h-pntOnScreen.y;

    float z_ndc = Renderer::GetPixelDepth(x, y, m_frameBuffer)*2-1;
    float z_cs = m_sight.DepthNDC2Clip(z_ndc);
    glm::vec4 pos_ndc(x/w*2-1.0, y/h*2-1.0, z_ndc, 1.0);
    glm::vec4 pos_cs = -z_cs*pos_ndc;
    glm::vec4 pos_ws = Clip2World()*pos_cs;

    return pos_ws;
}

glm::vec2 Camera::World2Screen(const glm::vec3& posInWorld)
{
    glm::vec4 pos_cs = World2Clip()*glm::vec4(posInWorld, 1.0);
    pos_cs /= pos_cs.w;
    return { (pos_cs.x+1)*m_windowSize[0]/2, m_windowSize[1]-(pos_cs.y+1)*m_windowSize[1]/2 };
}

glm::vec3 Camera::World2View(const glm::vec3& posInWorld)
{
    return World2View()*glm::vec4(posInWorld, 1);
}

glm::vec3 Camera::View2World(const glm::vec3& posInView)
{
    return View2World()*glm::vec4(posInView, 1);
}

void Camera::_ShowPosTooptips()
{
    ImVec2 screenpnt = ImGui::GetIO().MousePos;
    ImGui::SetNextWindowPos(ImVec2(screenpnt.x+10, screenpnt.y+10));
    ImGui::Begin("1", NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground);
    auto [x, y] = Input::GetMousePosition();
    glm::vec3 pos_ws1 = Screen2World({x, y});
    ImGui::Text("screenpnt: (%.0f, %.0f)", x, y);

    unsigned int w = m_windowSize[0];
    unsigned int h = m_windowSize[1];
    y = h-y;
    float z_ndc = Renderer::GetPixelDepth(x, y, m_frameBuffer)*2-1;
    float z_cs = m_sight.DepthNDC2Clip(z_ndc);
    glm::vec4 pos_ndc(x/w*2-1.0, y/h*2-1.0, z_ndc, 1.0);
    ImGui::Text("       ndc: (%.3f, %.3f, %.6f, %.3f)",  pos_ndc.x, pos_ndc.y, pos_ndc.z, pos_ndc.w);
    glm::vec4 pos_cs = -z_cs*pos_ndc;
    ImGui::Text(" clipspace: (%.3f, %.3f, %.3f, %.3f)",  pos_cs.x, pos_cs.y, pos_cs.z, pos_cs.w);
    glm::vec4 pos_vs = m_sight.Clip2View()*pos_cs;
    ImGui::Text(" viewspace: (%.3f, %.3f, %.3f, %.3f)",  pos_vs.x, pos_vs.y, pos_vs.z, pos_vs.w);
    glm::vec4 pos_ws = View2World()*pos_vs;
    ImGui::Text("worldspace: (%.3f, %.3f, %.3f, %.3f)",  pos_ws.x, pos_ws.y, pos_ws.z, pos_ws.w);
    ImGui::Text("worldspace: (%.3f, %.3f, %.3f)",  pos_ws1.x, pos_ws1.y, pos_ws1.z);
    glm::vec2 pnt = World2Screen(pos_ws);
    ImGui::Text("screenpnt: (%.0f, %.0f)", pnt.x, pnt.y);
    ImGui::End();
}

void Camera::OnImGuiRender(bool independent)
{
//     _ShowPosTooptips();
    if(independent)
    {
        ImGui::Begin(m_name.c_str());
    }
    else
    {
        ImGui::Text(m_name.c_str());
    }

    if(ImGui::Button("revert"))
    {
        Revert();
    }
    m_sight.OnImGuiRender();
    if(ImGui::RadioButton("2D", m_usage == Usage::TwoDimension) && m_usage != Usage::TwoDimension)
    {
        m_usage = Usage::TwoDimension;
        m_dirty = true;
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("3D", m_usage == Usage::ThreeDimension) && m_usage != Usage::ThreeDimension)
    {
        m_usage = Usage::ThreeDimension;
        m_dirty = true;
    }
    ImGui::Separator();

    ImGui::DragFloat("speed of translation", &m_speedTrans, 0.01f, 0, 0.8f);
    ImGui::DragFloat("speed of rotation", &m_speedRotat, 1.0f, 1.0f, 180.0f);
    ImGui::DragFloat("speed of scale", &m_speedScale, 0.05f, 0.01f, 20.0f);
    ImGui::Separator();
    ImGui::Checkbox("enable rotation", &m_rotationEnabled);
    ImGui::SameLine();
    ImGui::Checkbox("fix target", &m_targetFixed);
    ImGui::Separator();

    ImGui::Separator();
    m_dirty |= ImGui::SliderFloat3("position", (float*)glm::value_ptr(m_position), -1000, 1000);
    m_dirty |= ImGui::SliderFloat3("target", (float*)glm::value_ptr(m_target), -1000, 1000);
    m_dirty |= ImGui::SliderFloat3("up", (float*)glm::value_ptr(m_up), -1, 1);

    ImGui::Text("   position: %s",  glm::to_string(m_position).c_str());
    ImGui::Text("     target: %s",  glm::to_string(m_target).c_str());
    ImGui::Text("         up: %s",  glm::to_string(m_up).c_str());
    ImGui::Text("orientation: %s",  glm::to_string(m_orientation).c_str());
    ImGui::Text("      scale: %6.3f",  GetScale());

    if(independent)
    {
        ImGui::End();
    }
}
