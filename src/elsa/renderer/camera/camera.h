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
    inline void SetPosition(const glm::vec3& position) { m_position = position; m_dirty = true; }
    inline virtual void SetRotation(const glm::vec3& rotation) { m_rotation = rotation; m_dirty = true; }
    inline virtual void SetRotationX(float rx) { m_rotation.x = rx; } 
    inline virtual void SetRotationY(float ry) { m_rotation.y = ry; }
    inline virtual void SetRotationZ(float rz) { m_rotation.z = rz; }
    inline float GetRotationX() { return m_rotation.x; }
    inline float GetRotationY() { return m_rotation.y; } 
    inline float GetRotationZ() { return m_rotation.z; }

    inline void Translate(const glm::vec3& distance) { m_position += distance; m_dirty = true; }
    inline void Rotate(const glm::vec3& angle) { m_rotation += angle; m_dirty = true; }
    inline void Revert() { m_position = glm::vec3(0.0f); m_rotation = glm::vec3(0.0f); m_dirty = true; }
    inline const glm::mat4& GetViewMatrix() { _UpdateViewMatrix(); return m_matView; }
    inline const glm::mat4& GetProjectionMatrix() { return m_matProjection; }
    inline const glm::mat4 GetViewProjectionMatrix() { _UpdateViewMatrix(); return m_matViewProjection; }

    bool IsDirty() const { return m_dirty; }

protected:
    void _UpdateViewMatrix();
    bool m_dirty = true;

protected:
    glm::mat4 m_matView = glm::mat4(1.0f);
    glm::mat4 m_matProjection = glm::mat4(1.0f);
    glm::mat4 m_matViewProjection = glm::mat4(1.0f);

    glm::vec3 m_position = glm::vec3(0.0f);
    glm::vec3 m_rotation = glm::vec3(0.0f);
};
