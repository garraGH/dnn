/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : camera.h
* author      : Garra
* time        : 2019-10-03 20:18:41
* description : 
*
============================================*/


#pragma once
#include "glm/glm.hpp"

class Camera
{
public:
    Camera();
    virtual ~Camera();

    inline const glm::vec3& GetPosition() const { return m_position; }
    inline const glm::vec3& GetRotation() const { return m_rotation; }
    inline void SetPosition(const glm::vec3& position) { m_position = position; _UpdateViewMatrix(); }
    inline virtual void SetRotation(const glm::vec3& rotation) { m_rotation = rotation; _UpdateViewMatrix(); }
    inline virtual void SetRotationX(float rx) { m_rotation.x = rx; } 
    inline virtual void SetRotationY(float ry) { m_rotation.y = ry; }
    inline virtual void SetRotationZ(float rz) { m_rotation.z = rz; }
    inline float GetRotationX() { return m_rotation.x; }
    inline float GetRotationY() { return m_rotation.y; } 
    inline float GetRotationZ() { return m_rotation.z; }

    inline const glm::mat4& GetViewMatrix() const { return m_matView; }
    inline const glm::mat4& GetProjectionMatrix() const { return m_matProjection; }
    inline const glm::mat4 GetViewProjectionMatrix() const { return m_matViewProjection; }

protected:
    void _UpdateViewMatrix();

protected:
    glm::mat4 m_matView = glm::mat4(1.0f);
    glm::mat4 m_matProjection = glm::mat4(1.0f);
    glm::mat4 m_matViewProjection = glm::mat4(1.0f);

    glm::vec3 m_position = glm::vec3(0, 0, 0);
    glm::vec3 m_rotation = glm::vec3(0, 0, 0);
};
