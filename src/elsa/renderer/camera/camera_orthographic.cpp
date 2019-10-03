/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : camera_orthographic.cpp
* author      : Garra
* time        : 2019-10-03 20:18:42
* description : 
*
============================================*/



#include "camera_orthographic.h"
#include "glm/gtc/matrix_transform.hpp"

OrthographicCamera::OrthographicCamera(float left, float right, float bottom, float top, float near, float far)
{
    m_matProjection = glm::ortho(left, right, bottom, top, near, far);
    m_matViewProjection = m_matProjection*m_matView;
}

OrthographicCamera::~OrthographicCamera()
{

}
