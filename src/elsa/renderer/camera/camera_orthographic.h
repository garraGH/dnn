/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : camera_orthographic.h
* author      : Garra
* time        : 2019-10-03 20:18:41
* description : 
*
============================================*/


#pragma once
#include "camera.h"

class OrthographicCamera : public Camera
{
public:
    OrthographicCamera(float left, float right, float bottom, float top, float near=+1, float far=-1);
    virtual ~OrthographicCamera();

    virtual void SetRotation(const glm::vec3& rotation) override { m_rotation.z = rotation.z; _UpdateViewMatrix(); }
    virtual void SetRotationX(float rx) override {}
    virtual void SetRotationY(float ry) override {}
    virtual void SetRotationZ(float rz) override { m_rotation.z = rz; _UpdateViewMatrix(); }
    
};
