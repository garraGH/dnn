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
// #include "glm/gtx/string_cast.hpp"
// #include "logger.h"

Camera::Camera()
{

}

Camera::~Camera()
{

}

void Camera::_UpdateViewMatrix()
{
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), m_position) 
        * glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.x), glm::vec3(1, 0, 0)) 
        * glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.y), glm::vec3(0, 1, 0)) 
        * glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.z), glm::vec3(0, 0, -1));   // make positive rotate ccw
    
    m_matView = glm::inverse(transform);
    m_matViewProjection = m_matProjection*m_matView;

//     CORE_INFO("Camera::_UpdateViewMatrix");
//     CORE_INFO("View: {}", glm::to_string(m_matView));
//     CORE_INFO("Proj: {}", glm::to_string(m_matProjection));
//     CORE_INFO("ViPr: {}", glm::to_string(m_matViewProjection));
}
