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
// #include "camera_orthographic.h"
// #include "camera_perspective.h"
#include "imgui.h"
#include "glm/gtx/string_cast.hpp"
// #include "logger.h"
#include "../../core.h"
#include "../../input/input.h"

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
    if(ImGui::RadioButton("Orthographic", m_type == Type::Orthographic))
    {
        m_type = Type::Orthographic;
    }
    if(ImGui::RadioButton("Perspective", m_type == Type::Perspective))
    {
        m_type = Type::Perspective;
    }
    if(m_type == Type::Orthographic)
    {
        ImGui::SliderFloat("Width", &m_width, 0, 1000);
    }
    else
    {
        ImGui::SliderFloat("vFov", &m_vfov, 0, 180);
    }
    ImGui::SameLine();
    ImGui::DragFloat("scale", &m_scale,  0.1,  0.1,  10);
    ImGui::SameLine();
    ImGui::Text("asp: %.3f", m_asp);
    ImGui::DragFloat("near", &m_near,  0.1,  0.1, 10);
    ImGui::SameLine();
    ImGui::DragFloat("far", &m_far,  10,  100,  10000);
    m_dirty = true;

}

void Camera::_UpdateView()
{
    if(!m_dirty)
    {
        return;
    }
    m_dirty = false;
    glm::mat4 transform = glm::rotate(glm::mat4(1.0f), glm::radians(m_orientation.x), glm::vec3(1, 0, 0)) 
                        * glm::rotate(glm::mat4(1.0f), glm::radians(m_orientation.y), glm::vec3(0, 1, 0)) 
                        * glm::rotate(glm::mat4(1.0f), glm::radians(m_orientation.z), glm::vec3(0, 0, 1))   
                        * glm::translate(glm::mat4(1.0f), m_position);
    if(m_usage == Usage::TwoDimension)
    {
        transform = glm::rotate(glm::mat4(1.0f), glm::radians(m_orientation.z), glm::vec3(0, 0, 1)) * glm::translate(glm::mat4(1.0f), m_position);
    }

    m_matView = glm::inverse(transform);
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
}

bool Camera::_OnMouseScrolled(MouseScrolledEvent& e)
{
    Scale(-e.GetOffsetY()*m_speedScale);
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
    if(ImGui::RadioButton("Orthographic", m_type == Type::Orthographic))
    {
        m_type = Type::Orthographic;
        m_sight.SetType(m_type);
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("Perspective", m_type == Type::Perspective))
    {
        m_type = Type::Perspective;
        m_sight.SetType(m_type);
    }
    if(ImGui::RadioButton("2D", m_usage == Usage::TwoDimension))
    {
        m_usage = Usage::TwoDimension;
        m_dirty = true;
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("3D", m_usage == Usage::ThreeDimension))
    {
        m_usage = Usage::ThreeDimension;
        m_dirty = true;
    }
    ImGui::Separator();

    ImGui::DragFloat("speed of translation", &m_speedTrans,  0.05f,  0.01f,  2.0f);
    ImGui::DragFloat("speed of rotation", &m_speedRotat,  1.0f,  1.0f,  180.0f);
    ImGui::DragFloat("speed of scale", &m_speedScale,  0.02f,  0.01f,  20.0f);
    ImGui::Separator();
    ImGui::Checkbox("enable rotation", &m_rotationEnabled);
    ImGui::SameLine();
    ImGui::Checkbox("fix target", &m_targetFixed);
    ImGui::Separator();

    ImGui::Separator();
    ImGui::Text("   position: %s",  glm::to_string(m_position).c_str());
    ImGui::Text("     target: %s",  glm::to_string(m_target).c_str());
    ImGui::Text("  direction: %s",  glm::to_string(m_direction).c_str());
    ImGui::Text("         up: %s",  glm::to_string(m_up).c_str());
    ImGui::Text("orientation: %s",  glm::to_string(m_orientation).c_str());
    ImGui::Text("      scale: %6.3f",  GetScale());

    if(independent)
    {
        ImGui::End();
    }
}
