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
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

class OrthographicCamera : public Camera
{
public:
    OrthographicCamera() { m_type = Type::Orthographic; }

    virtual void SetRotation(const glm::vec3& rotation) override { m_rotation.z = rotation.z; m_poseDirty = true; }
    virtual void SetRotationX(float rx) override {}
    virtual void SetRotationY(float ry) override {}
    virtual void SetRotationZ(float rz) override { m_rotation.z = rz; m_poseDirty = true; }

protected:
    virtual void _UpdateAsp() { m_matProjection = glm::ortho(-m_aspectRatio*m_scale, +m_aspectRatio*m_scale, -m_scale, +m_scale, +1.0f, -1.0f); m_aspDirty = false; }
};
