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
}

void Camera::Sight::_calcOrthoMatrix()
{
    m_matProjection[0][0] = 2.0/m_width/m_scale;
    m_matProjection[1][1] = 2.0*m_asp/m_width/m_scale;
    m_matProjection[2][2] = -2.0/(m_far-m_near);
    m_matProjection[2][3] = 0;
    m_matProjection[3][2] = -(m_far+m_near)/(m_far-m_near);
    m_matProjection[3][3] = 1;
}

void Camera::Sight::_calcPerspectiveMatrix()
{
    m_matProjection[1][1] = 1.0/glm::tan(glm::radians(m_vfov)*0.5)/m_scale;
    m_matProjection[0][0] = m_matProjection[1][1]/m_asp;
    m_matProjection[2][2] = -(m_far+m_near)/(m_far-m_near);
    m_matProjection[2][3] = -1;
    m_matProjection[3][2] = -2.0*m_far*m_near/(m_far-m_near);
    m_matProjection[3][3] = 0;
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

void Camera::_UpdateView()
{
    if(!m_dirty)
    {
        return;
    }
    m_dirty = false;
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
    m_matView = rot*trans;
    m_matWorld = glm::inverse(m_matView);
}
    
void Camera::_UpdateViewProjection()
{
    if(!NeedUpdate())
    {
        return;
    }

    _UpdateView();
    m_matViewProjection = m_sight.GetProjectionMatrix()*m_matView;
}

std::array<float, 12> Camera::GetCornersDirection() const
{
    float y = std::tan(glm::radians(m_sight.GetVfov()*0.5));
    float x = m_sight.GetAsp()*y;
    glm::vec3 pos0 = m_matWorld*glm::vec4(-x, -y, 1, 1);
    glm::vec3 dir0 = glm::normalize(m_position-pos0);
    glm::vec3 pos1 = m_matWorld*glm::vec4(+x, -y, 1, 1);
    glm::vec3 dir1 = glm::normalize(m_position-pos1);
    glm::vec3 pos2 = m_matWorld*glm::vec4(+x, +y, 1, 1);
    glm::vec3 dir2 = glm::normalize(m_position-pos2);
    glm::vec3 pos3 = m_matWorld*glm::vec4(-x, +y, 1, 1);
    glm::vec3 dir3 = glm::normalize(m_position-pos3);
    return {dir0.x, dir0.y, dir0.z, dir1.x, dir1.y, dir1.z, dir2.x, dir2.y, dir2.z, dir3.x, dir3.y, dir3.z};
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
        Translate(glm::vec3(0, 0, -e.GetOffsetY()*m_speedTrans));
    }
    return false;
}

bool Camera::_OnMouseButtonPressed(MouseButtonPressedEvent& e)
{
    m_bLeftButtonPressed = e.GetMouseButton() == MOUSE_BUTTON_LEFT ;
    m_bRightButtonPressed = e.GetMouseButton() == MOUSE_BUTTON_RIGHT;
    m_cursorPosPressed = Input::GetMousePosition();
    m_cursorPosPrevious = m_cursorPosPressed;
    return false;
}

#define PI 3.14159265
bool Camera::_OnMouseMoved(MouseMovedEvent& e)
{
    if(!(m_bLeftButtonPressed||m_bRightButtonPressed))
    {
        return false;
    }

    m_dirty = true;

    float xCurr = e.GetX();
    float yCurr = e.GetY();

    float dx = +e.GetX()-m_cursorPosPrevious.first;
    float dy = -e.GetY()+m_cursorPosPrevious.second;

    m_cursorPosPrevious = {xCurr, yCurr};


    if(m_bRightButtonPressed)
    {
        dx *= PI/m_windowSize[0];
        dy *= PI/m_windowSize[1];
        glm::vec3 dir = m_target-m_position;
        dir.y = 0;
        float r = glm::length(dir);
        float theta = std::atan2(dir.z, dir.x);
        theta += dx;
        if(Input::IsKeyPressed(KEY_LEFT_CONTROL))
        {
            m_position.x = m_target.x-r*std::cos(theta);
            m_position.z = m_target.z-r*std::sin(theta);
        }
        else
        {
            m_target.x = m_position.x+r*std::cos(theta);
            m_target.z = m_position.z+r*std::sin(theta);
        }

        dir = m_target-m_position;
        dir.x = 0;
        r = glm::length(dir);
        theta = std::atan2(dir.z, dir.y);
        theta += dy;
        if(Input::IsKeyPressed(KEY_LEFT_CONTROL))
        {
            m_position.y = m_target.y-r*std::cos(theta);
            m_position.z = m_target.z-r*std::sin(theta);
        }
        else
        {
            m_target.y = m_position.y+r*std::cos(theta);
            m_target.z = m_position.z+r*std::sin(theta);
        }

//         m_up.y = (m_position.y>=m_target.y)? +1 : -1;

    }
    if(m_bLeftButtonPressed)
    {
        dx *= 100.0/m_windowSize[0];
        dy *= 100.0/m_windowSize[1];
        m_position += glm::vec3(dx, dy, 0);
        if(Input::IsKeyPressed(KEY_LEFT_CONTROL))
        {
            m_target += glm::vec3(dx, dy, 0);
        }
    }
    return false;
}

bool Camera::_OnMouseButtonReleased(MouseButtonReleasedEvent& e)
{
    m_bLeftButtonPressed &= !(e.GetMouseButton() == MOUSE_BUTTON_LEFT);
    m_bRightButtonPressed &= !(e.GetMouseButton() == MOUSE_BUTTON_RIGHT);
    return false;
}

bool Camera::_OnWindowResize(WindowResizeEvent& e)
{
    m_windowSize = {e.GetWidth(), e.GetHeight()};
    return false;
}

void Camera::OnImGuiRender(bool independent)
{
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

    ImGui::DragFloat("speed of translation", &m_speedTrans, 1, 0.1f, 100.0f);
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
