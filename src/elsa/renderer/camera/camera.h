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
#include "memory"
#include "glm/glm.hpp"
#include "../../event/event_key.h"
#include "../../event/event_mouse.h"
#include "../../event/event_application.h"

class Camera
{
public:
    enum class Type
    {
        Unknown, 
        Orthographic, 
        Perspective 
    };

    
    Camera();
    virtual ~Camera();

    
    inline const glm::vec3& GetTranslation() const { return m_translation; }
    inline const glm::vec3& GetRotation() const { return m_rotation; }
    inline void SetTranslation(const glm::vec3& translation) { m_translation = translation; m_poseDirty = true; }
    inline virtual void SetRotation(const glm::vec3& rotation) { m_rotation = rotation; m_poseDirty = true; }
    inline virtual void SetRotationX(float rx) { m_rotation.x = rx; } 
    inline virtual void SetRotationY(float ry) { m_rotation.y = ry; }
    inline virtual void SetRotationZ(float rz) { m_rotation.z = rz; }

    inline void SetAspectRatio(float asp) { m_aspectRatio = asp; m_aspDirty = true; }
    inline float GetRotationX() { return m_rotation.x; }
    inline float GetRotationY() { return m_rotation.y; } 
    inline float GetRotationZ() { return m_rotation.z; }

    inline void Translate(const glm::vec3& dis) { m_translation += dis; m_poseDirty = true; }
    inline void Rotate(const glm::vec3& angle) { m_rotation += angle; m_poseDirty = true; }
    inline void Scale(float s) { m_scale += s; m_poseDirty = true; }
    inline void Revert() { m_translation = glm::vec3(0.0f); m_rotation = glm::vec3(0.0f); m_scale = 1.0f; m_poseDirty = true; }
    inline const glm::mat4& GetViewMatrix() { _Update(); return m_matView; }
    inline const glm::mat4& GetProjectionMatrix() { _Update(); return m_matProjection; }
    inline const glm::mat4 GetViewProjectionMatrix() { _Update(); return m_matViewProjection; }

    inline bool IsDirty() const { return m_aspDirty || m_poseDirty; }

    static std::shared_ptr<Camera> Create(Type type);

    friend class CameraContoller;
protected:
    inline bool _NotDirty() const { return !IsDirty(); }
    void _Update();
    void _UpdatePose();
    virtual void _UpdateAsp() {};
    void _UpdateViewProjection();


protected:
    bool m_aspDirty = true;
    bool m_poseDirty = true;
    Type m_type = Type::Unknown;
    float m_aspectRatio = 1.0f;

    glm::mat4 m_matView = glm::mat4(1.0f);
    glm::mat4 m_matProjection = glm::mat4(1.0f);
    glm::mat4 m_matViewProjection = glm::mat4(1.0f);

    glm::vec3 m_translation = glm::vec3(0.0f);
    glm::vec3 m_rotation = glm::vec3(0.0f);
    float m_scale = 1.0f;
};


class CameraContoller
{
public:
    CameraContoller(Camera::Type type);

    void SetAspectRatio(float asp) { m_camera->SetAspectRatio(asp); }
    void EnableRotation(bool enabled) { m_rotationEnabled = enabled; }

    void SetTranslationSpeed(float speed) { m_speedTrans = speed; }
    void SetRotationSpeed(float speed) { m_speedRotat = speed; }
    void SetScaleSpeed(float speed) { m_speedScale = speed; }

    void OnUpdate(float deltaTime);
    void OnEvent(Event& e);
    void Revert() { m_camera->Revert(); }
    void OnImGuiRender();

    const std::shared_ptr<Camera>& GetCamera() const { return m_camera; }

protected:
    bool _OnMouseScrolled(MouseScrolledEvent& e);
    bool _OnWindowResized(WindowResizeEvent& e);

private:
    bool m_rotationEnabled = true;
    float m_speedTrans = 0.5f;
    float m_speedRotat = 20.0f;
    float m_speedScale = 0.1f;
    std::shared_ptr<Camera> m_camera;

};
