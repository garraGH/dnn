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
#include "camera_orthographic.h"
#include "camera_perspective.h"
#include "imgui.h"
#include "glm/gtx/string_cast.hpp"
// #include "logger.h"
#include "../../core.h"
#include "../../input/input.h"

std::shared_ptr<Camera> Camera::Create(Type type)
{
    switch(type)
    {
        case Type::Orthographic: return std::make_shared<OrthographicCamera>();
        case Type::Perspective: return std::make_shared<PerspectiveCamera>();
        default: CORE_ASSERT(false, "Unknown Camera Type!"); return nullptr;
    }
}

Camera::Camera()
    : m_type(Type::Unknown)
{
    
}

Camera::~Camera()
{

}

void Camera::_Update()
{
    if(_NotDirty())
    {
        return;
    }

    _UpdateAsp();
    _UpdatePose();
    _UpdateViewProjection();
}

void Camera::_UpdatePose()
{
    if(!m_poseDirty)
    {
        return;
    }

    glm::mat4 transform = glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.x), glm::vec3(1, 0, 0)) 
                        * glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.y), glm::vec3(0, 1, 0)) 
                        * glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.z), glm::vec3(0, 0, -1))   // make positive rotate ccw
                        * glm::translate(glm::mat4(1.0f), m_translation);
    m_matView = glm::inverse(transform);
    m_poseDirty = false;
}
    
void Camera::_UpdateViewProjection()
{
    m_matViewProjection = m_matProjection*m_matView;
}


//////////////////////////////////////////
//
CameraContoller::CameraContoller(Camera::Type type)
{
    m_camera = Camera::Create(type);
}

void CameraContoller::OnUpdate(float deltaTime)
{
    if(Input::IsKeyPressed(KEY_S))
    {
        m_camera->Translate({+deltaTime*m_speedTrans, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_F))
    {
        m_camera->Translate({-deltaTime*m_speedTrans, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_W))
    {
        m_camera->Translate({0, +deltaTime*m_speedTrans, 0});
    }
    if(Input::IsKeyPressed(KEY_R))
    {
        m_camera->Translate({0, -deltaTime*m_speedTrans, 0});
    }
    if(Input::IsKeyPressed(KEY_E))
    {
        m_camera->Scale(-deltaTime*m_speedScale);
    }
    if(Input::IsKeyPressed(KEY_D))
    {
        m_camera->Scale(+deltaTime*m_speedScale);
    }

    if(m_rotationEnabled)
    {
        if(Input::IsKeyPressed(KEY_K))
        {
            m_camera->Rotate({0, 0, -deltaTime*m_speedRotat});
        }
        if(Input::IsKeyPressed(KEY_J))
        {
            m_camera->Rotate({0, 0, +deltaTime*m_speedRotat});
        }
    }
}


void CameraContoller::OnEvent(Event& e)
{
    EventDispatcher dispatcher(e);
    dispatcher.Dispatch<MouseScrolledEvent>(BIND_EVENT_CALLBACK(CameraContoller, _OnMouseScrolled));
    dispatcher.Dispatch<WindowResizeEvent>(BIND_EVENT_CALLBACK(CameraContoller, _OnWindowResized));
}

bool CameraContoller::_OnMouseScrolled(MouseScrolledEvent& e)
{
    m_camera->Scale(-e.GetOffsetY()*m_speedScale);
    return false;
}


bool CameraContoller::_OnWindowResized(WindowResizeEvent& e)
{
    m_camera->SetAspectRatio((float)e.GetWidth()/e.GetHeight());
    return false;
}

void CameraContoller::OnImGuiRender()
{
    ImGui::Begin("CameraContoller");

    ImGui::DragFloat("speed of translation", &m_speedTrans,  0.05f,  0.01f,  2.0f);
    ImGui::DragFloat("speed of rotation", &m_speedRotat,  1.0f,  1.0f,  180.0f);
    ImGui::DragFloat("speed of scale", &m_speedScale,  0.02f,  0.01f,  20.0f);
    ImGui::Checkbox("enable rotation", &m_rotationEnabled);
    if(ImGui::Button("revert"))
    {
        Revert();
    }
    ImGui::Text("translation: %s",  glm::to_string(m_camera->GetTranslation()).c_str());
    ImGui::Text("   rotation: %s",  glm::to_string(m_camera->GetRotation()).c_str());
    ImGui::Text("      scale: %6.3f",  m_camera->GetScale());

    ImGui::End();
}
